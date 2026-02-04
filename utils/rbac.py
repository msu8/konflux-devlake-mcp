#!/usr/bin/env python3
"""
Role-Based Access Control (RBAC) for Konflux DevLake MCP Server

This module provides email-based authorization for MCP tools.
Roles are mapped to allowed tools, enforcing the principle of least privilege.

Role Assignment (email-based):
- If user's email is in RBAC_ADMIN_EMAILS -> mcp-admin (full access incl. execute_query)
- Otherwise -> mcp-viewer (all tools EXCEPT execute_query)

Configure via environment variables (set in OCP ConfigMap):
  - RBAC_ADMIN_EMAILS: Comma-separated list of admin email addresses
  - RBAC_DEFAULT_ROLE: Role for users not in admin list (default: mcp-viewer)
"""

import os
from typing import Any, Dict, List, Optional, Set

from utils.logger import get_logger

# Role definitions mapping Keycloak groups to allowed tools
# These group names should match what's configured in Red Hat SSO / Keycloak
ROLE_PERMISSIONS: Dict[str, Set[str]] = {
    # Viewer role - read-only access to analytics and reports
    # For employees pulling stats and viewing dashboards
    "mcp-viewer": {
        # Schema exploration
        "connect_database",
        "list_databases",
        "list_tables",
        "get_table_schema",
        # Incident analysis
        "get_incidents",
        "get_failed_deployment_recovery_time",
        # Deployment metrics
        "get_deployments",
        "get_deployment_frequency",
        # PR analytics
        "analyze_pr_retests",
        "get_pr_cycle_time",
        "get_pr_stats",
        # CI/CD health
        "get_github_actions_health",
        "analyze_e2e_tests",
        "get_codecov_coverage",
        "get_codecov_summary",
        # Metrics and trends
        "get_historical_trends",
        "get_lead_time_for_changes",
        # Jira
        "get_jira_features",
    },
    # Admin role - full access including raw SQL queries
    # For managers and MCP administrators
    "mcp-admin": {"*"},  # Wildcard means all tools
}

# Default role for authenticated users without specific group assignment
DEFAULT_ROLE: Optional[str] = None  # None means no access without explicit role

# Sentinel value to distinguish "not provided" from "explicitly set to None"
_NOT_SET = object()


def get_admin_emails_from_env() -> Set[str]:
    """
    Get admin email addresses from environment variable.

    The RBAC_ADMIN_EMAILS environment variable should contain a comma-separated
    list of email addresses that should have admin access.

    Returns:
        Set of admin email addresses (lowercase for case-insensitive matching)
    """
    env_value = os.environ.get("RBAC_ADMIN_EMAILS", "")
    if not env_value:
        return set()

    emails = {email.strip().lower() for email in env_value.split(",") if email.strip()}
    return emails


def get_default_role_from_env() -> Optional[str]:
    """
    Get default role from environment variable.

    Returns:
        Default role name, or None if not set
    """
    return os.environ.get("RBAC_DEFAULT_ROLE", "mcp-viewer")


class AuthorizationService:
    """
    Authorization service for enforcing role-based access control.

    This service checks if a user is authorized to call specific MCP tools
    based on their email address.

    Role Resolution (email-based only):
    - If user's email is in RBAC_ADMIN_EMAILS (ConfigMap) -> mcp-admin (full access)
    - Otherwise -> mcp-viewer (no execute_query)
    """

    def __init__(
        self,
        role_permissions: Optional[Dict[str, Set[str]]] = None,
        default_role: Any = _NOT_SET,
        admin_emails: Optional[Set[str]] = None,
        use_email_roles: bool = True,
    ):
        """
        Initialize the authorization service.

        Args:
            role_permissions: Custom role-to-permissions mapping (uses default if None)
            default_role: Default role for users without explicit group assignment.
                          Pass None to disable default roles, or omit to use env var.
            admin_emails: Set of email addresses that should have admin access
            use_email_roles: Whether to use email-based role assignment (default: True)
        """
        self.logger = get_logger(f"{__name__}.AuthorizationService")
        self.role_permissions = role_permissions or ROLE_PERMISSIONS

        # Handle default_role: _NOT_SET means read from env, None means no default
        if default_role is _NOT_SET:
            self.default_role = get_default_role_from_env()
        else:
            self.default_role = default_role

        self.use_email_roles = use_email_roles

        # Get admin emails from parameter or environment
        if admin_emails is not None:
            self.admin_emails = {e.lower() for e in admin_emails}
        else:
            self.admin_emails = get_admin_emails_from_env()

        self.logger.info(
            f"Authorization service initialized with {len(self.role_permissions)} roles, "
            f"{len(self.admin_emails)} admin emails, default_role={self.default_role}"
        )
        if self.admin_emails:
            self.logger.info(f"Admin emails configured: {len(self.admin_emails)} addresses")

    def resolve_user_roles(
        self, user_groups: List[str], user_email: Optional[str] = None
    ) -> List[str]:
        """
        Resolve the effective roles for a user based on email.

        Role assignment is purely email-based:
        - Email in admin list (RBAC_ADMIN_EMAILS) -> mcp-admin
        - Otherwise -> mcp-viewer (default)

        Args:
            user_groups: Groups from the user's OIDC token (not used, kept for interface)
            user_email: User's email address from OIDC token

        Returns:
            List of resolved role names
        """
        # Email-based role assignment only
        if user_email:
            email_lower = user_email.lower()
            if email_lower in self.admin_emails:
                self.logger.info(f"Admin role assigned via email whitelist: {user_email}")
                return ["mcp-admin"]

        # Default role for everyone else
        if self.default_role:
            self.logger.debug(f"Default role '{self.default_role}' assigned to {user_email}")
            return [self.default_role]

        return []

    def is_authorized(
        self,
        user_groups: List[str],
        tool_name: str,
        user_email: Optional[str] = None,
    ) -> bool:
        """
        Check if a user is authorized to call a specific tool.

        Args:
            user_groups: List of groups/roles from the user's OIDC token
            tool_name: Name of the tool being called
            user_email: Optional email for email-based role resolution

        Returns:
            True if the user is authorized, False otherwise
        """
        # Resolve effective roles (considering groups, email, and defaults)
        effective_roles = self.resolve_user_roles(user_groups, user_email)

        if not effective_roles:
            self.logger.warning(f"Access denied: no roles resolved for tool '{tool_name}'")
            return False

        # Check each resolved role
        for role in effective_roles:
            if self._role_allows_tool(role, tool_name):
                self.logger.debug(f"Access granted: role '{role}' allows tool '{tool_name}'")
                return True

        self.logger.warning(
            f"Access denied: roles {effective_roles} not authorized for tool '{tool_name}'"
        )
        return False

    def _role_allows_tool(self, role: str, tool_name: str) -> bool:
        """
        Check if a specific role allows access to a tool.

        Args:
            role: Role/group name
            tool_name: Tool name to check

        Returns:
            True if the role allows the tool
        """
        allowed_tools = self.role_permissions.get(role, set())

        # Check for wildcard (admin access)
        if "*" in allowed_tools:
            return True

        return tool_name in allowed_tools

    def get_allowed_tools(
        self, user_groups: List[str], user_email: Optional[str] = None
    ) -> Set[str]:
        """
        Get all tools a user is allowed to call based on their roles.

        Args:
            user_groups: List of groups/roles from the user's OIDC token
            user_email: Optional email for email-based role resolution

        Returns:
            Set of allowed tool names (or {"*"} for full access)
        """
        # Resolve effective roles
        effective_roles = self.resolve_user_roles(user_groups, user_email)

        allowed = set()

        # Add tools from each resolved role
        for role in effective_roles:
            role_tools = self.role_permissions.get(role, set())
            if "*" in role_tools:
                return {"*"}  # Full access
            allowed.update(role_tools)

        return allowed

    def get_denied_reason(
        self, user_groups: List[str], tool_name: str, user_email: Optional[str] = None
    ) -> str:
        """
        Get a human-readable reason why access was denied.

        Args:
            user_groups: User's groups
            tool_name: Tool that was denied
            user_email: User's email (for more informative message)

        Returns:
            Explanation string for the denial
        """
        # Resolve effective roles to show what they actually have
        effective_roles = self.resolve_user_roles(user_groups, user_email)

        if not effective_roles:
            return (
                f"Access denied: no authorized roles found. "
                f"Tool '{tool_name}' requires one of: {list(self.role_permissions.keys())}"
            )

        # Find which roles would allow this tool
        required_roles = []
        for role, tools in self.role_permissions.items():
            if "*" in tools or tool_name in tools:
                required_roles.append(role)

        if required_roles:
            return (
                f"Access denied: your roles {effective_roles} do not include "
                f"permissions for tool '{tool_name}'. "
                f"Required role(s): {required_roles}"
            )

        return f"Access denied: tool '{tool_name}' is not available in any role"

    def get_role_info(self) -> Dict[str, Any]:
        """
        Get information about configured roles for debugging/monitoring.

        Returns:
            Dictionary with role configuration info
        """
        role_info = {}
        for role, tools in self.role_permissions.items():
            if "*" in tools:
                role_info[role] = {"access": "full", "tools": ["*"]}
            else:
                role_info[role] = {"access": "limited", "tools": sorted(list(tools))}

        return {
            "roles": role_info,
            "default_role": self.default_role,
            "total_roles": len(self.role_permissions),
            "email_based_roles": self.use_email_roles,
            "admin_emails_configured": len(self.admin_emails),
        }
