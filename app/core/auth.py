from dataclasses import dataclass

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer

from app.core.config import settings
from app.services.database_service import get_user_id_from_access_token


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str


# Fixed placeholder row seeded into dbo.users by database/mssql_schema.sql.
# Used below while authentication is disabled, so scrape_jobs/api_cost_logs
# still get a user_id that satisfies their foreign key to dbo.users.
ANONYMOUS_USER_ID = "00000000-0000-0000-0000-000000000000"


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/token",
    refreshUrl="/token/refresh",
    auto_error=False,
    description=(
        "Supabase email/password login. In Swagger, enter your email in the "
        "username field; the returned access token is added automatically."
    ),
)


# API authentication is disabled -- both dependencies below always succeed.
# To re-enable, uncomment the original bodies and remove the bypass returns.
def require_authenticated_user(
    access_token: str | None = Security(oauth2_scheme),
) -> AuthenticatedUser:
    return AuthenticatedUser(user_id=ANONYMOUS_USER_ID)
    # if not access_token:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Missing Authorization bearer token.",
    #         headers={"WWW-Authenticate": "Bearer"},
    #     )
    # user_id = get_user_id_from_access_token(access_token)
    # return AuthenticatedUser(user_id=user_id)


def require_developer_user(
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> AuthenticatedUser:
    return user
    # developer_user_ids = {
    #     user_id.strip()
    #     for user_id in settings.developer_user_ids.split(",")
    #     if user_id.strip()
    # }
    # if not developer_user_ids:
    #     raise HTTPException(
    #         status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    #         detail="Developer access is not configured.",
    #     )
    # if user.user_id not in developer_user_ids:
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="Developer access is required.",
    #     )
    # return user
