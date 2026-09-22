from typing import Annotated

from fastapi import APIRouter, Form, Response

from app.schemas.auth import AccessTokenResponse
from app.services.database_service import (
    AuthSessionTokens,
    authenticate_user_with_password,
    refresh_access_token,
)

router = APIRouter(tags=["authentication"])


def _token_response(tokens: AuthSessionTokens) -> AccessTokenResponse:
    return AccessTokenResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        token_type=tokens.token_type,
        expires_in=tokens.expires_in,
    )


def _prevent_token_caching(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"


@router.post(
    "/token",
    response_model=AccessTokenResponse,
    summary="Log in and get a Supabase access token",
)
def login_for_access_token(
    response: Response,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    grant_type: Annotated[str | None, Form(pattern="^password$")] = None,
) -> AccessTokenResponse:
    """Use the OAuth2 username field for the user's Supabase email address."""
    del grant_type
    tokens = authenticate_user_with_password(
        email=username,
        password=password,
    )
    _prevent_token_caching(response)
    return _token_response(tokens)


@router.post(
    "/token/refresh",
    response_model=AccessTokenResponse,
    summary="Refresh a Supabase access token",
)
def refresh_tokens(
    response: Response,
    grant_type: Annotated[str, Form(pattern="^refresh_token$")],
    refresh_token: Annotated[str, Form(min_length=1)],
) -> AccessTokenResponse:
    del grant_type
    tokens = refresh_access_token(refresh_token)
    _prevent_token_caching(response)
    return _token_response(tokens)
