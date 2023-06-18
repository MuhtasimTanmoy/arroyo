use arroyo_server_common::log_event;
use axum::extract::State;
use axum::headers::authorization::{Authorization, Bearer};
use axum::response::{IntoResponse, Response};
use axum::{http::StatusCode, routing::post, Json, Router, TypedHeader};
use deadpool_postgres::{Object, Pool};
use serde_json::json;
use tracing::error;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::connections::{
    ConnectionTypes, HttpConnection, KafkaAuthConfig, KafkaConnection, PostConnections, SaslAuth,
};
use crate::{cloud, connections, AuthData};

type BearerAuth = Option<TypedHeader<Authorization<Bearer>>>;

#[derive(OpenApi)]
#[openapi(
    info(title = "Arroyo REST API", version = "1.0.0"),
    paths(create_connection),
    components(schemas(
        PostConnections,
        ConnectionTypes,
        KafkaConnection,
        KafkaAuthConfig,
        SaslAuth,
        HttpConnection
    )),
    tags(
        (name = "connection", description = "Connections management API")
    )
)]
struct ApiDoc;

#[derive(Clone)]
struct AppState {
    pool: Pool,
}

pub struct ErrorResp {
    pub(crate) status_code: StatusCode,
    pub(crate) message: String,
}

pub fn log_and_map_rest<E>(err: E) -> ErrorResp
where
    E: core::fmt::Debug,
{
    error!("Error while handling: {:?}", err);
    log_event("api_error", json!({ "error": format!("{:?}", err) }));
    ErrorResp {
        status_code: StatusCode::INTERNAL_SERVER_ERROR,
        message: "Something went wrong".to_string(),
    }
}

impl IntoResponse for ErrorResp {
    fn into_response(self) -> Response {
        let body = Json(json!({
            "error": self.message,
        }));
        (self.status_code, body).into_response()
    }
}

async fn client(pool: &Pool) -> Result<Object, ErrorResp> {
    pool.get().await.map_err(log_and_map_rest)
}

async fn authenticate(pool: &Pool, bearer_auth: BearerAuth) -> Result<AuthData, ErrorResp> {
    let client = client(pool).await?;
    cloud::authenticate_rest(client, bearer_auth).await
}

#[utoipa::path(
    post,
    path = "/v1/connections",
    tag = "connection",
    request_body = PostConnections,
    responses(
        (status = 200, description = "Connection created successfully", body = PostConnections),
    ),
)]
async fn create_connection(
    State(state): State<AppState>,
    bearer_auth: BearerAuth,
    Json(payload): Json<PostConnections>,
) -> Result<(), ErrorResp> {
    let auth_data = authenticate(&state.pool, bearer_auth).await?;
    let client = client(&state.pool).await?;
    let connection = payload.clone().into();
    connections::create_connection(connection, auth_data, client).await
}

pub(crate) fn create_rest_app(pool: Pool) -> Router {
    Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/v1/connections", post(create_connection))
        .with_state(AppState { pool })
}