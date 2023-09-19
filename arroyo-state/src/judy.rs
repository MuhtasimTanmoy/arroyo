use std::io::SeekFrom;
use std::collections::BTreeMap;

use std::convert::TryFrom;

use anyhow::{bail, Result};
use tokio::io::{AsyncReadExt, AsyncSeek, AsyncSeekExt, AsyncWriteExt};

pub struct JudyNode {
    children: BTreeMap<u8, JudyNode>,
    offset_contribution: Option<u32>,
    // these are bytes that are common to the terminal node and all children.
    // when traversing, append these to the key you're building up.
    common_bytes: Vec<u8>,
    terminal_offset: Option<u32>,
    is_root: bool,
}

pub struct JudyWriter {
    judy_node: JudyNode,
    bytes_written: usize,
    data_buffer: Vec<u8>,
}

impl JudyWriter {
    pub fn new() -> Self {
        JudyWriter {
            judy_node: JudyNode::new(),
            bytes_written: 0,
            data_buffer: Vec::new(),
        }
    }

    pub async fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        let offset = self.bytes_written.try_into()?;
        self.bytes_written += 4 + value.len();
        // TODO: not use 4 full bytes every time.
        self.data_buffer
            .write_u32_le(value.len().try_into()?)
            .await?;
        self.data_buffer.write_all(value).await?;
        self.judy_node.insert(key, offset);
        Ok(())
    }

    pub async fn serialize<W: tokio::io::AsyncWrite + Unpin + std::marker::Send>(
        &mut self,
        writer: &mut W,
    ) -> Result<()> {
        self.judy_node.serialize(writer).await?;
        writer.write_all(&self.data_buffer).await?;
        writer.flush().await?;
        Ok(())
    }
}

impl JudyNode {
    pub fn new() -> Self {
        JudyNode {
            children: BTreeMap::new(),
            offset_contribution: None,
            common_bytes: Vec::new(),
            terminal_offset: None,
            is_root: true,
        }
    }
    pub fn insert(&mut self, key: &[u8], offset: u32) {
        if key.is_empty() {
            self.terminal_offset = Some(offset);
            return;
        }

        self.children
            .entry(key[0])
            .or_insert_with(|| JudyNode {
                children: BTreeMap::new(),
                offset_contribution: None,
                common_bytes: Vec::new(),
                terminal_offset: None,
                is_root: false,
            })
            .insert(&key[1..], offset);
    }

    // return the earliest data in the node and subnodes.
    // this should be an absolute value, and the min of terminal_offset.
    fn earliest_data(&self) -> u32 {
        match self.terminal_offset {
            Some(offset) => offset,
            None => {
                let Some(first_child) = self.children.values().next() else {
                    unreachable!("non-terminal JudyNode with no children");
                };
                first_child.earliest_data()
            }
        }
    }

    fn optimize_offsets(&mut self, parent_offset: u32) {
        let node_data_start = self.earliest_data();
        self.offset_contribution = Some(node_data_start - parent_offset);
        for child in self.children.values_mut() {
            child.optimize_offsets(node_data_start);
        }
    }

    fn compress_nodes(&mut self) {
        for child in self.children.values_mut() {
            child.compress_nodes();
        }
        if self.children.len() == 1 && self.terminal_offset.is_none() {
            // this is a node that doesn't have data and one child, so merge it with its child.
            let (trie_byte, child_node) = self.children.pop_first().unwrap();
            self.common_bytes.push(trie_byte);
            self.common_bytes.extend(child_node.common_bytes);
            self.children = child_node.children;
            self.terminal_offset = child_node.terminal_offset;
        } else if self.terminal_offset.is_none() && !self.children.is_empty() {
            // this node doesn't have data, so we can move the offset contribution up to the parent.
            if let Some(min_child_offset) = self
                .children
                .values()
                .map(|child| child.offset_contribution.unwrap_or_default())
                .min()
            {
                self.offset_contribution =
                    Some(min_child_offset + self.offset_contribution.unwrap_or_default());
                self.children.values_mut().for_each(|child| {
                    let new_offset =
                        child.offset_contribution.unwrap_or_default() - min_child_offset;
                    if new_offset > 0 {
                        child.offset_contribution = Some(new_offset);
                    } else {
                        child.offset_contribution = None;
                    }
                })
            }
        }
    }
    pub async fn serialize<W: tokio::io::AsyncWrite + Unpin + std::marker::Send>(
        &mut self,
        writer: &mut W,
    ) -> Result<()> {
        self.compress_nodes();
        self.optimize_offsets(0);

        self.serialize_internal(writer).await?;
        writer.flush().await?;
        Ok(())
    }

    #[async_recursion::async_recursion]
    async fn serialize_internal<W: tokio::io::AsyncWrite + Unpin + std::marker::Send>(
        &self,
        writer: &mut W,
    ) -> Result<()> {
        let serialization_stats = self.serialization_stats();
        let offset_contribution = if self.is_root {
            if self.offset_contribution.unwrap_or_default() > 0 {
                bail!("Root node should have 0 as offset contribution");
            }
            Some(serialization_stats.total_size as u32)
        } else {
            self.offset_contribution
        };
        let header = serialization_stats.header;

        writer.write_u8(header.to_byte()).await?;
        if !self.common_bytes.is_empty() {
            writer.write(&[self.common_bytes.len() as u8]).await?;
            writer.write_all(&self.common_bytes).await?;
        }
        header
            .offset_contribution_size
            .write(offset_contribution, writer)
            .await?;
        if self.children.is_empty() {
            return Ok(());
        }
        header
            .children_density
            .write_keys(serialization_stats.keys, writer)
            .await?;
        header
            .write_pointers(serialization_stats.map_offsets, writer)
            .await?;
        for child in self.children.values() {
            if !child.is_simple_leaf() {
                child.serialize_internal(writer).await?;
            }
        }
        Ok(())
    }

    fn serialization_stats(&self) -> SerializationStats {
        let mut size = 1; // 1 byte for the flag

        let has_common_bytes = if self.common_bytes.is_empty() {
            HasCommonBytes::No
        } else {
            HasCommonBytes::Yes
        };

        size += has_common_bytes.byte_count();
        size += self.common_bytes.len();

        // special case for root so we can set the initial location of the data.
        let offset_contribution_size = if self.is_root {
            VariableByteSize::FourBytes
        } else {
            match self.offset_contribution {
                Some(val) if val <= u8::MAX as u32 => VariableByteSize::OneByte,
                Some(val) if 2 * val + 1 <= u16::MAX as u32 => VariableByteSize::TwoBytes,
                Some(_) => VariableByteSize::FourBytes,
                None => VariableByteSize::None,
            }
        };

        size += offset_contribution_size.byte_count();
        let has_terminal_value = if self.terminal_offset.is_some() {
            HasTerminalValue::Yes
        } else {
            HasTerminalValue::No
        };

        // if you don't have children. This should only happen if you've compressed a leaf node,
        // in which case the separate node is needed to store the common_bytes.
        if self.children.is_empty() {
            return SerializationStats::new(
                size,
                JudyNodeHeader::new(
                    offset_contribution_size,
                    has_terminal_value,
                    ChildrenDensity::Sparse,
                    VariableByteSize::None,
                    has_common_bytes,
                ),
                vec![],
                vec![],
            );
        }
        let children_density = if self.children.len() <= 16 {
            size += 1 + self.children.len();
            ChildrenDensity::Sparse
        } else {
            size += 32;
            ChildrenDensity::Dense
        };

        let mut offsets = vec![];
        let mut children_size = 0;
        for (key, child) in &self.children {
            if child.is_simple_leaf() {
                offsets.push((
                    *key,
                    child.offset_contribution.unwrap_or_default() as usize,
                    true,
                ));
            } else {
                let child_size = child.serialization_stats().total_size;
                offsets.push((*key, children_size, false));
                children_size += child_size;
            }
        }
        let largest_offset = offsets
            .iter()
            .map(|(_, offset, _)| offset)
            .max()
            .unwrap()
            .clone();

        // need to figure out how many bytes we need to encode the largest offset.
        // the index offsets will be relative to the index we're currently creating, so need to take that into account.
        let mut child_pointer_size = VariableByteSize::FourBytes;
        size += child_pointer_size.byte_count() * offsets.len();
        if largest_offset + size < i16::MAX as usize {
            size -= child_pointer_size.byte_count() * offsets.len();
            child_pointer_size = VariableByteSize::TwoBytes;
            size += child_pointer_size.byte_count() * offsets.len();
            if largest_offset + size < i8::MAX as usize {
                size -= child_pointer_size.byte_count() * offsets.len();
                child_pointer_size = VariableByteSize::OneByte;
                size += child_pointer_size.byte_count() * offsets.len();
            }
        }
        // right now size is of the data for the specific node. We can now compute the values to actually write to the map.
        let (keys, map_values): (Vec<_>, Vec<_>) = offsets
            .iter()
            .map(|(key, offset, is_leaf)| {
                if *is_leaf {
                    let offset = (*offset as i32) * -1;
                    (*key, offset)
                } else {
                    (*key, (*offset as i32) + size as i32)
                }
            })
            .unzip();
        SerializationStats::new(
            size + children_size,
            JudyNodeHeader::new(
                offset_contribution_size,
                has_terminal_value,
                children_density,
                child_pointer_size,
                has_common_bytes,
            ),
            keys,
            map_values,
        )
    }

    fn is_simple_leaf(&self) -> bool {
        self.children.is_empty() && self.common_bytes.is_empty()
    }

    #[async_recursion::async_recursion]
    pub async fn get_data<R: tokio::io::AsyncRead + Unpin + Send + AsyncSeek>(
        index_reader: &mut R,
        data_reader: &mut R,
        mut key: Vec<u8>,
    ) -> Result<Option<Vec<u8>>> {
        let starting_position = index_reader.stream_position().await?;

        let mut header_byte = [0u8; 1];
        index_reader.read_exact(&mut header_byte).await?;
        let header = JudyNodeHeader::try_from(header_byte[0])?;

        if header.has_common_bytes == HasCommonBytes::Yes {
            let mut common_len = [0u8; 1];
            index_reader.read_exact(&mut common_len).await?;
            let mut common_bytes = vec![0u8; common_len[0] as usize];
            index_reader.read_exact(&mut common_bytes).await?;
            if !key.starts_with(&common_bytes) {
                return Ok(None);
            }
            key = key[common_bytes.len()..].to_vec();
        }
        let offset_contribution = header
            .offset_contribution_size
            .read_u32(index_reader)
            .await?;
        if let Some(offset_contribution) = offset_contribution {
            data_reader
                .seek(SeekFrom::Current(offset_contribution as i64))
                .await?;
        }
        if key.is_empty() {
            if header.has_terminal_value() {
                let value_size = data_reader.read_u32_le().await?;
                let mut value = vec![0u8; value_size as usize];
                data_reader.read_exact(&mut value).await?;
                return Ok(Some(value));
            } else {
                return Ok(None);
            }
        }
        let next_byte = key[0];
        key = key[1..].to_vec();
        let Some(child_offset) = header.get_child_offset(next_byte, index_reader).await? else {
            return Ok(None);
        };
        if child_offset <= 0 {
            data_reader
                .seek(SeekFrom::Current(-child_offset as i64))
                .await?;
            let value_size = data_reader.read_u32_le().await?;
            let mut value = vec![0u8; value_size as usize];
            data_reader.read_exact(&mut value).await?;
            Ok(Some(value))
        } else {
            index_reader
                .seek(SeekFrom::Start(starting_position + child_offset as u64))
                .await?;
            Self::get_data(index_reader, data_reader, key).await
        }
    }
}

struct SerializationStats {
    total_size: usize,
    header: JudyNodeHeader,
    keys: Vec<u8>,
    map_offsets: Vec<i32>,
}

impl SerializationStats {
    fn new(
        total_size: usize,
        header: JudyNodeHeader,
        keys: Vec<u8>,
        map_offsets: Vec<i32>,
    ) -> Self {
        SerializationStats {
            total_size,
            header,
            keys,
            map_offsets,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum VariableByteSize {
    None = 0b00,
    OneByte = 0b01,
    TwoBytes = 0b10,
    FourBytes = 0b11,
}
impl VariableByteSize {
    fn byte_count(&self) -> usize {
        match self {
            VariableByteSize::None => 0,
            VariableByteSize::OneByte => 1,
            VariableByteSize::TwoBytes => 2,
            VariableByteSize::FourBytes => 4,
        }
    }

    async fn write<W: tokio::io::AsyncWrite + Unpin>(
        &self,
        offset_contribution: Option<u32>,
        writer: &mut W,
    ) -> Result<()> {
        match self {
            VariableByteSize::None => {}
            VariableByteSize::OneByte => {
                writer.write_u8(offset_contribution.unwrap() as u8).await?;
            }
            VariableByteSize::TwoBytes => {
                writer
                    .write_u16_le(offset_contribution.unwrap() as u16)
                    .await?;
            }
            VariableByteSize::FourBytes => {
                writer.write_u32_le(offset_contribution.unwrap()).await?;
            }
        }
        Ok(())
    }

    async fn read_u32<R: tokio::io::AsyncRead + Unpin>(
        &self,
        reader: &mut R,
    ) -> Result<Option<u32>> {
        match self {
            VariableByteSize::None => Ok(None),
            VariableByteSize::OneByte => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf).await?;
                Ok(Some(buf[0] as u32))
            }
            VariableByteSize::TwoBytes => {
                let mut buf = [0u8; 2];
                reader.read_exact(&mut buf).await?;
                Ok(Some(u16::from_le_bytes(buf) as u32))
            }
            VariableByteSize::FourBytes => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf).await?;
                Ok(Some(u32::from_le_bytes(buf)))
            }
        }
    }
    async fn read_i32<R: tokio::io::AsyncRead + Unpin>(
        &self,
        reader: &mut R,
    ) -> Result<Option<i32>> {
        match self {
            VariableByteSize::None => Ok(None),
            VariableByteSize::OneByte => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf).await?;
                Ok(Some((buf[0] as i8) as i32))
            }
            VariableByteSize::TwoBytes => {
                let mut buf = [0u8; 2];
                reader.read_exact(&mut buf).await?;
                Ok(Some(i16::from_le_bytes(buf) as i32))
            }
            VariableByteSize::FourBytes => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf).await?;
                Ok(Some(i32::from_le_bytes(buf)))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum HasTerminalValue {
    No = 0,
    Yes = 1,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum ChildrenDensity {
    Sparse = 0,
    Dense = 1,
}
impl ChildrenDensity {
    async fn write_keys<W: tokio::io::AsyncWrite + Unpin>(
        &self,
        keys: Vec<u8>,
        writer: &mut W,
    ) -> Result<()> {
        match self {
            ChildrenDensity::Sparse => {
                writer.write_u8((keys.len() - 1) as u8).await?;
                for key in keys {
                    writer.write_u8(key).await?;
                }
            }
            ChildrenDensity::Dense => {
                let mut bitmap = vec![0; 32];
                for key in keys {
                    let byte_index = key / 8;
                    let bit_index = key % 8;
                    bitmap[byte_index as usize] |= 1 << bit_index;
                }
                writer.write_all(&bitmap).await?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum HasCommonBytes {
    No = 0,
    Yes = 1,
}
impl HasCommonBytes {
    fn byte_count(&self) -> usize {
        match self {
            HasCommonBytes::No => 0,
            HasCommonBytes::Yes => 1,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct JudyNodeHeader {
    offset_contribution_size: VariableByteSize,
    has_terminal_value: HasTerminalValue,
    children_density: ChildrenDensity,
    child_pointer_size: VariableByteSize,
    has_common_bytes: HasCommonBytes,
}

impl JudyNodeHeader {
    fn new(
        offset_contribution_size: VariableByteSize,
        has_terminal_value: HasTerminalValue,
        children_density: ChildrenDensity,
        child_pointer_size: VariableByteSize,
        has_common_bytes: HasCommonBytes,
    ) -> Self {
        JudyNodeHeader {
            offset_contribution_size,
            has_terminal_value,
            children_density,
            child_pointer_size,
            has_common_bytes,
        }
    }

    // Convert the struct into a single byte flag
    fn to_byte(&self) -> u8 {
        let mut flag = 0u8;

        flag |= (self.offset_contribution_size as u8) & 0b11; // Bits 0-1
        flag |= (self.has_terminal_value as u8) << 2; // Bit 2
        flag |= (self.children_density as u8) << 3; // Bit 3
        flag |= (self.child_pointer_size as u8) << 4; // Bits 4-5
        flag |= (self.has_common_bytes as u8) << 6; // Bit 6

        flag
    }

    fn has_terminal_value(&self) -> bool {
        self.has_terminal_value == HasTerminalValue::Yes
    }

    async fn get_child_offset<R: tokio::io::AsyncRead + Unpin + AsyncSeek>(
        &self,
        byte: u8,
        reader: &mut R,
    ) -> Result<Option<i32>> {
        match self.children_density {
            ChildrenDensity::Sparse => {
                let key_count = (reader.read_u8().await? as usize) + 1;
                // scan key_count, looking for bytes
                for i in 0..key_count {
                    let key = reader.read_u8().await?;
                    if key == byte {
                        reader
                            .seek(SeekFrom::Current(
                                ((key_count - i - 1) + self.child_pointer_size.byte_count() * i)
                                    as i64,
                            ))
                            .await?;
                        return Ok(self.child_pointer_size.read_i32(reader).await?);
                    } else if byte < key {
                        return Ok(None);
                    }
                }
                Ok(None)
            }
            ChildrenDensity::Dense => {
                // advance key/8
                let mut bitmap = [0u8; 32];
                reader.read_exact(&mut bitmap).await?;
                // if the bit is set, it will be the Nth offset
                let byte_index = (byte / 8) as usize;
                let bit_index = byte % 8;
                if bitmap[byte_index] & (1 << bit_index) == 0 {
                    return Ok(None);
                }
                let mut rank = 0;
                for i in 0..byte_index {
                    rank += bitmap[i as usize].count_ones() as usize;
                }
                rank += (bitmap[byte_index] & ((1 << bit_index) - 1)).count_ones() as usize;
                reader
                    .seek(SeekFrom::Current(
                        (rank * self.child_pointer_size.byte_count()) as i64,
                    ))
                    .await?;
                Ok(self.child_pointer_size.read_i32(reader).await?)
            }
        }
    }

    async fn write_pointers<W: tokio::io::AsyncWrite + Unpin>(
        &self,
        map_offsets: Vec<i32>,
        writer: &mut W,
    ) -> Result<()> {
        match self.child_pointer_size {
            VariableByteSize::OneByte => {
                for offset in map_offsets {
                    writer.write_i8(offset as i8).await?;
                }
            }
            VariableByteSize::TwoBytes => {
                for offset in map_offsets {
                    writer.write_i16_le(offset as i16).await?;
                }
            }
            VariableByteSize::FourBytes => {
                for offset in map_offsets {
                    writer.write_i32_le(offset as i32).await?;
                }
            }
            VariableByteSize::None => {
                bail!("shouldn't be writing pointers without a valid byte size")
            }
        }
        Ok(())
    }
}

impl TryFrom<u8> for JudyNodeHeader {
    type Error = anyhow::Error; // You can use a custom Error type here

    fn try_from(value: u8) -> Result<Self> {
        let offset_contribution_size = match value & 0b11 {
            0 => VariableByteSize::None,
            1 => VariableByteSize::OneByte,
            2 => VariableByteSize::TwoBytes,
            3 => VariableByteSize::FourBytes,
            _ => bail!("Invalid offset contribution size".to_string()),
        };

        let has_terminal_value = match (value >> 2) & 1 {
            0 => HasTerminalValue::No,
            1 => HasTerminalValue::Yes,
            _ => bail!("Invalid terminal value flag".to_string()),
        };

        let children_density = match (value >> 3) & 1 {
            0 => ChildrenDensity::Sparse,
            1 => ChildrenDensity::Dense,
            _ => bail!("Invalid children density flag".to_string()),
        };

        let child_pointer_size = match (value >> 4) & 0b11 {
            0 => VariableByteSize::None,
            1 => VariableByteSize::OneByte,
            2 => VariableByteSize::TwoBytes,
            3 => VariableByteSize::FourBytes,
            _ => bail!("Invalid child pointer size".to_string()),
        };

        let has_common_bytes = match (value >> 6) & 1 {
            0 => HasCommonBytes::No,
            1 => HasCommonBytes::Yes,
            _ => bail!("Invalid common bytes flag".to_string()),
        };

        Ok(JudyNodeHeader {
            offset_contribution_size,
            has_terminal_value,
            children_density,
            child_pointer_size,
            has_common_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        io::{Cursor, SeekFrom},
    };

    use super::{JudyNode, JudyWriter};
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    // Adjust the import as necessary
    use anyhow::Result;
    use tokio::io::AsyncSeekExt;

    struct TestHarness {
        writer: JudyWriter,
        serialized_data: Vec<u8>,
    }

    impl TestHarness {
        pub fn new() -> Self {
            Self {
                writer: JudyWriter::new(),
                serialized_data: Vec::new(),
            }
        }

        pub async fn add_data(&mut self, key: &[u8], data: &[u8]) -> Result<()> {
            self.writer.insert(key, data).await?;
            Ok(())
        }

        pub async fn serialize_files(&mut self) -> Result<()> {
            self.writer.serialize(&mut self.serialized_data).await?;
            Ok(())
        }

        pub async fn test_retrieval(
            &self,
            key: Vec<u8>,
            expected_data: Option<Vec<u8>>,
        ) -> Result<()> {
            let mut cursor = Cursor::new(self.serialized_data.clone());
            cursor.seek(SeekFrom::Start(0)).await?;
            let result = JudyNode::get_data(
                &mut cursor,
                &mut Cursor::new(self.serialized_data.clone()),
                key,
            )
            .await?;
            assert_eq!(result, expected_data);
            Ok(())
        }
    }

    #[tokio::test]
    async fn judy_node_test() -> Result<()> {
        let mut harness = TestHarness::new();

        // Add data to the test
        harness
            .add_data(
                vec![0x6b, 0x65, 0x79, 0x31].as_slice(),
                vec![1, 2, 3, 4].as_slice(),
            )
            .await?;
        harness
            .add_data(
                vec![0x6b, 0x65, 0x79, 0x32].as_slice(),
                vec![3, 4, 5, 6, 7].as_slice(),
            )
            .await?;

        // Serialize files
        harness.serialize_files().await?;

        // Test retrieval
        harness
            .test_retrieval(vec![0x6b, 0x65, 0x79, 0x31], Some(vec![1, 2, 3, 4]))
            .await?;
        harness
            .test_retrieval(vec![0x6b, 0x65, 0x79, 0x32], Some(vec![3, 4, 5, 6, 7]))
            .await?;
        harness
            .test_retrieval(vec![0x6b, 0x65, 0x79, 0x33], None)
            .await?;

        Ok(())
    }
    async fn run_random_test(
        depth: usize,
        density: f32,
        data_size: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut harness = TestHarness::new(); // Add constructor or initialization logic
        let mut expected_map = BTreeMap::new();
        let mut rng = SmallRng::seed_from_u64(0);

        // Use depth, density, and data_size to generate your JudyNode
        // Example: Varying keys based on 'depth' and 'density'
        let total_keys = ((1 << depth) as f32 * density).ceil() as usize;

        for _ in 0..total_keys {
            let key_len: usize = rng.gen_range(1..depth);
            let key: Vec<u8> = (0..key_len).map(|_| rng.gen()).collect();

            let data: Vec<u8> = (0..data_size).map(|_| rng.gen()).collect();

            expected_map.insert(key, data);
        }
        let mut i = 0;
        for (key, value) in &expected_map {
            i += 1;
            harness.add_data(key, value).await?;
        }

        harness.serialize_files().await?;

        i = 0;
        // Validate the JudyNode
        for (key, expected_data) in &expected_map {
            harness
                .test_retrieval(key.clone(), Some(expected_data.clone()))
                .await
                .expect(&format!("failed on key {}", i));
            i += 1;
        }

        Ok(())
    }
    #[tokio::test]
    async fn variable_judy_node_test() -> Result<()> {
        run_random_test(5, 0.5, 32).await.unwrap();
        run_random_test(10, 0.7, 64).await.unwrap();
        run_random_test(7, 0.9, 16).await.unwrap();
        run_random_test(24, 0.01, 32).await.unwrap();
        Ok(())
    }
}
