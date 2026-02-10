#!/usr/bin/env python3
"""
Unit Tests for Role-Based Access Control (RBAC)

Tests for utils/rbac.py and RBAC integration in tool_handler.py
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from utils.rbac import (
    AuthorizationService,
    ROLE_PERMISSIONS,
    get_admin_usernames_from_env,
    get_default_role_from_env,
    extract_username_from_email,
)
from server.handlers.tool_handler import (
    ToolHandler,
    set_user_context,
    get_user_context,
)


class TestAuthorizationService:
    """Tests for AuthorizationService class."""

    def test_initialization_with_defaults(self):
        """Test authorization service initializes with default roles and viewer as default."""
        auth_service = AuthorizationService()

        assert auth_service.role_permissions == ROLE_PERMISSIONS
        # Default role is mcp-viewer (from get_default_role_from_env)
        assert auth_service.default_role == "mcp-viewer"
        assert auth_service.use_email_roles is True

    def test_initialization_with_strict_mode(self):
        """Test authorization service with strict mode (no default role)."""
        auth_service = AuthorizationService(default_role=None, use_email_roles=False)

        assert auth_service.default_role is None
        assert auth_service.use_email_roles is False

    def test_initialization_with_custom_roles(self):
        """Test authorization service with custom role definitions."""
        custom_roles = {
            "custom-viewer": {"list_databases"},
            "custom-admin": {"*"},
        }

        auth_service = AuthorizationService(
            role_permissions=custom_roles, default_role=None, use_email_roles=False
        )

        assert auth_service.role_permissions == custom_roles

    def test_viewer_can_access_allowed_tools(self):
        """Test that mcp-viewer role can access read-only tools."""
        auth_service = AuthorizationService()

        # Viewer should be able to access these tools
        allowed_tools = [
            "list_databases",
            "list_tables",
            "get_table_schema",
            "get_incidents",
            "get_deployments",
            "get_pr_stats",
            "get_historical_trends",
        ]

        for tool in allowed_tools:
            assert auth_service.is_authorized(["mcp-viewer"], tool) is True

    def test_viewer_cannot_access_execute_query(self):
        """Test that mcp-viewer role cannot access execute_query."""
        auth_service = AuthorizationService()

        # Viewer should NOT be able to execute raw queries
        assert auth_service.is_authorized(["mcp-viewer"], "execute_query") is False

    def test_admin_can_access_all_tools(self):
        """Test that mcp-admin role can access all tools including execute_query."""
        auth_service = AuthorizationService(admin_usernames={"admin"})

        # Admin should be able to access everything
        all_tools = [
            "list_databases",
            "execute_query",
            "get_incidents",
            "some_future_tool",  # Wildcard should allow any tool
        ]

        for tool in all_tools:
            # Use email-based admin assignment
            assert auth_service.is_authorized([], tool, user_email="admin@example.com") is True

    def test_user_with_no_groups_denied_strict(self):
        """Test that users without groups are denied access in strict mode."""
        # Strict mode: no default role, no email-based roles
        auth_service = AuthorizationService(default_role=None, use_email_roles=False)

        assert auth_service.is_authorized([], "list_databases") is False

    def test_user_with_no_groups_gets_default_role(self):
        """Test that users without groups get default viewer role."""
        # Default behavior: users get mcp-viewer role
        auth_service = AuthorizationService()

        # Should be able to access viewer tools
        assert auth_service.is_authorized([], "list_databases") is True
        # But not admin tools
        assert auth_service.is_authorized([], "execute_query") is False

    def test_user_with_unknown_group_denied_strict(self):
        """Test that users with unknown groups are denied access in strict mode."""
        auth_service = AuthorizationService(default_role=None, use_email_roles=False)

        assert auth_service.is_authorized(["unknown-group"], "list_databases") is False

    def test_user_with_unknown_group_gets_default_role(self):
        """Test that users with unknown groups get default viewer role."""
        auth_service = AuthorizationService()

        # Should be able to access viewer tools (default role applied)
        assert auth_service.is_authorized(["unknown-group"], "list_databases") is True
        # But not admin tools
        assert auth_service.is_authorized(["unknown-group"], "execute_query") is False

    def test_user_with_multiple_groups(self):
        """Test user with multiple groups gets combined permissions."""
        auth_service = AuthorizationService()

        # User has both viewer and some unknown group
        groups = ["unknown-group", "mcp-viewer"]

        # Should still have viewer access
        assert auth_service.is_authorized(groups, "get_incidents") is True
        # But not admin access
        assert auth_service.is_authorized(groups, "execute_query") is False

    def test_admin_email_grants_full_access(self):
        """Test that admin email grants full access including execute_query."""
        auth_service = AuthorizationService(admin_usernames={"admin"})

        # Admin email should grant full access
        assert (
            auth_service.is_authorized([], "execute_query", user_email="admin@example.com") is True
        )
        # Non-admin email should not have execute_query
        assert (
            auth_service.is_authorized([], "execute_query", user_email="user@example.com") is False
        )

    def test_get_allowed_tools_for_viewer(self):
        """Test getting allowed tools for viewer role."""
        auth_service = AuthorizationService()

        allowed = auth_service.get_allowed_tools(["mcp-viewer"])

        assert "list_databases" in allowed
        assert "get_incidents" in allowed
        assert "execute_query" not in allowed
        assert "*" not in allowed

    def test_get_allowed_tools_for_admin(self):
        """Test getting allowed tools for admin role returns wildcard."""
        auth_service = AuthorizationService(admin_usernames={"admin"})

        # Use email-based admin assignment
        allowed = auth_service.get_allowed_tools([], user_email="admin@example.com")

        assert allowed == {"*"}

    def test_get_denied_reason_no_groups_strict(self):
        """Test denied reason message for user with no groups in strict mode."""
        auth_service = AuthorizationService(default_role=None, use_email_roles=False)

        reason = auth_service.get_denied_reason([], "execute_query")

        assert "no authorized roles found" in reason
        assert "execute_query" in reason

    def test_get_denied_reason_no_groups_default(self):
        """Test denied reason when user gets default role but tool is admin-only."""
        auth_service = AuthorizationService()  # Default: mcp-viewer role

        reason = auth_service.get_denied_reason([], "execute_query")

        # User has mcp-viewer role (default), but can't access execute_query
        assert "mcp-viewer" in reason
        assert "execute_query" in reason
        assert "mcp-admin" in reason  # Should mention the required role

    def test_get_denied_reason_wrong_role(self):
        """Test denied reason message for user with insufficient role."""
        auth_service = AuthorizationService()

        reason = auth_service.get_denied_reason(["mcp-viewer"], "execute_query")

        assert "mcp-viewer" in reason
        assert "execute_query" in reason
        assert "mcp-admin" in reason  # Should mention the required role

    def test_get_role_info(self):
        """Test getting role configuration info."""
        auth_service = AuthorizationService()

        info = auth_service.get_role_info()

        assert "roles" in info
        assert "mcp-viewer" in info["roles"]
        assert "mcp-admin" in info["roles"]
        assert info["roles"]["mcp-admin"]["access"] == "full"
        assert info["roles"]["mcp-viewer"]["access"] == "limited"

    def test_default_role_assignment(self):
        """Test default role for users without explicit assignment."""
        auth_service = AuthorizationService(default_role="mcp-viewer", use_email_roles=False)

        # User with no groups should get default role permissions
        assert auth_service.is_authorized([], "get_incidents") is True
        assert auth_service.is_authorized([], "execute_query") is False


class TestUsernameBasedRoles:
    """Tests for username-based role assignment."""

    def test_admin_username_gets_admin_role(self):
        """Test that admin username whitelist grants admin access."""
        admin_usernames = {"admin", "team-lead"}
        auth_service = AuthorizationService(admin_usernames=admin_usernames)

        # Admin username should get full access
        assert (
            auth_service.is_authorized([], "execute_query", user_email="admin@redhat.com") is True
        )
        assert (
            auth_service.is_authorized([], "list_databases", user_email="admin@redhat.com") is True
        )

    def test_non_admin_username_gets_viewer_role(self):
        """Test that non-admin username gets viewer role."""
        admin_usernames = {"admin"}
        auth_service = AuthorizationService(admin_usernames=admin_usernames)

        # Non-admin username should get viewer access only
        assert (
            auth_service.is_authorized([], "list_databases", user_email="user@redhat.com") is True
        )
        assert (
            auth_service.is_authorized([], "execute_query", user_email="user@redhat.com") is False
        )

    def test_username_matching_case_insensitive(self):
        """Test that username matching is case-insensitive."""
        admin_usernames = {"admin"}
        auth_service = AuthorizationService(admin_usernames=admin_usernames)

        # Should match regardless of case
        assert (
            auth_service.is_authorized([], "execute_query", user_email="ADMIN@REDHAT.COM") is True
        )
        assert (
            auth_service.is_authorized([], "execute_query", user_email="Admin@Redhat.Com") is True
        )

    def test_username_extracted_correctly(self):
        """Test that username is extracted from email correctly."""
        admin_usernames = {"daturece"}
        auth_service = AuthorizationService(admin_usernames=admin_usernames)

        # Username should be extracted from various email formats
        assert (
            auth_service.is_authorized([], "execute_query", user_email="daturece@redhat.com")
            is True
        )
        assert (
            auth_service.is_authorized([], "execute_query", user_email="daturece@example.org")
            is True
        )

    def test_resolve_user_roles_with_username(self):
        """Test resolving roles when username matches admin list."""
        admin_usernames = {"admin"}
        auth_service = AuthorizationService(admin_usernames=admin_usernames)

        roles = auth_service.resolve_user_roles([], "admin@redhat.com")
        assert "mcp-admin" in roles

    def test_resolve_user_roles_without_match(self):
        """Test resolving roles when username doesn't match admin list."""
        admin_usernames = {"admin"}
        auth_service = AuthorizationService(admin_usernames=admin_usernames)

        roles = auth_service.resolve_user_roles([], "user@redhat.com")
        assert "mcp-viewer" in roles
        assert "mcp-admin" not in roles

    def test_get_admin_usernames_from_env(self):
        """Test reading admin usernames from environment variable."""
        with patch.dict(os.environ, {"RBAC_ADMIN_USERNAMES": "admin1, admin2, Admin3"}):
            usernames = get_admin_usernames_from_env()

            assert "admin1" in usernames
            assert "admin2" in usernames
            assert "admin3" in usernames  # Should be lowercased

    def test_get_admin_usernames_empty_env(self):
        """Test reading admin usernames when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if it exists
            os.environ.pop("RBAC_ADMIN_USERNAMES", None)
            usernames = get_admin_usernames_from_env()

            assert usernames == set()

    def test_extract_username_from_email(self):
        """Test extracting username from email address."""
        assert extract_username_from_email("daturece@redhat.com") == "daturece"
        assert extract_username_from_email("ADMIN@EXAMPLE.COM") == "admin"
        assert extract_username_from_email("user") == "user"

    def test_get_default_role_from_env(self):
        """Test reading default role from environment."""
        # Default value when not set
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RBAC_DEFAULT_ROLE", None)
            role = get_default_role_from_env()
            assert role == "mcp-viewer"

        # Custom value
        with patch.dict(os.environ, {"RBAC_DEFAULT_ROLE": "custom-role"}):
            role = get_default_role_from_env()
            assert role == "custom-role"


class TestUserContext:
    """Tests for user context management."""

    def test_set_and_get_user_context(self):
        """Test setting and getting user context."""
        user_info = {
            "id": "user-123",
            "username": "testuser",
            "groups": ["mcp-viewer"],
        }

        set_user_context(user_info)
        retrieved = get_user_context()

        assert retrieved == user_info

    def test_clear_user_context(self):
        """Test clearing user context."""
        set_user_context({"id": "test"})
        set_user_context(None)

        assert get_user_context() is None

    def test_default_context_is_none(self):
        """Test that default context is None."""
        # Clear any existing context
        set_user_context(None)

        assert get_user_context() is None


class TestToolHandlerRBAC:
    """Tests for RBAC integration in ToolHandler."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_tools_manager = MagicMock()
        self.mock_security_manager = MagicMock()
        # Clear user context before each test
        set_user_context(None)

    def test_tool_handler_with_rbac_enabled(self):
        """Test tool handler initializes with RBAC enabled."""
        handler = ToolHandler(
            self.mock_tools_manager,
            self.mock_security_manager,
            rbac_enabled=True,
        )

        assert handler.rbac_enabled is True
        assert handler.authorization_service is not None

    def test_tool_handler_with_rbac_disabled(self):
        """Test tool handler initializes with RBAC disabled."""
        handler = ToolHandler(
            self.mock_tools_manager,
            self.mock_security_manager,
            rbac_enabled=False,
        )

        assert handler.rbac_enabled is False
        assert handler.authorization_service is None

    @pytest.mark.asyncio
    async def test_authorized_user_can_call_tool(self):
        """Test that authorized user can call tool successfully."""
        handler = ToolHandler(
            self.mock_tools_manager,
            self.mock_security_manager,
            rbac_enabled=True,
        )

        # Set user context with viewer role
        set_user_context(
            {
                "id": "user-123",
                "username": "viewer-user",
                "groups": ["mcp-viewer"],
            }
        )

        # Mock tool execution
        self.mock_tools_manager.call_tool.return_value = '{"success": true}'

        # Call a tool that viewer can access
        result = await handler.handle_tool_call("get_incidents", {})

        # Should succeed
        assert len(result) == 1
        self.mock_tools_manager.call_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_unauthorized_user_denied(self):
        """Test that unauthorized user is denied access."""
        handler = ToolHandler(
            self.mock_tools_manager,
            self.mock_security_manager,
            rbac_enabled=True,
        )

        # Set user context with viewer role
        set_user_context(
            {
                "id": "user-123",
                "username": "viewer-user",
                "groups": ["mcp-viewer"],
            }
        )

        # Try to call execute_query (admin only)
        result = await handler.handle_tool_call("execute_query", {"query": "SELECT 1"})

        # Should be denied
        assert len(result) == 1
        response_text = result[0].text
        assert "Access denied" in response_text
        # Tool should not be called
        self.mock_tools_manager.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_admin_can_call_any_tool(self):
        """Test that admin (via username whitelist) can call any tool."""
        # Create handler with admin usernames configured
        with patch.dict(os.environ, {"RBAC_ADMIN_USERNAMES": "admin"}):
            handler = ToolHandler(
                self.mock_tools_manager,
                self.mock_security_manager,
                rbac_enabled=True,
            )

        # Set user context with admin email
        set_user_context(
            {
                "id": "admin-123",
                "username": "admin-user",
                "email": "admin@example.com",
                "groups": [],
            }
        )

        # Mock tool execution
        self.mock_tools_manager.call_tool.return_value = '{"success": true}'
        self.mock_security_manager.validate_sql_query.return_value = (True, "OK")

        # Call execute_query (admin only)
        await handler.handle_tool_call("execute_query", {"query": "SELECT 1"})

        # Should succeed - tool was called
        self.mock_tools_manager.call_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_user_context_denied(self):
        """Test that missing user context results in denied access."""
        handler = ToolHandler(
            self.mock_tools_manager,
            self.mock_security_manager,
            rbac_enabled=True,
        )

        # No user context set
        set_user_context(None)

        # Try to call any tool
        result = await handler.handle_tool_call("list_databases", {})

        # Should be denied
        assert len(result) == 1
        response_text = result[0].text
        assert "authentication required" in response_text
        self.mock_tools_manager.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_rbac_disabled_allows_all(self):
        """Test that with RBAC disabled, all tools are accessible."""
        handler = ToolHandler(
            self.mock_tools_manager,
            self.mock_security_manager,
            rbac_enabled=False,
        )

        # No user context
        set_user_context(None)

        # Mock tool execution
        self.mock_tools_manager.call_tool.return_value = '{"success": true}'
        self.mock_security_manager.validate_sql_query.return_value = (True, "OK")

        # Call execute_query - should work without auth
        await handler.handle_tool_call("execute_query", {"query": "SELECT 1"})

        # Should succeed (no RBAC check) - tool was called
        self.mock_tools_manager.call_tool.assert_called_once()
