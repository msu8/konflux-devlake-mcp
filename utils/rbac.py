#!/usr/bin/env python3
"""
Role-Based Access Control (RBAC) for Konflux DevLake MCP Server

This module provides LDAP/Rover group-based authorization for MCP tools.
Roles are mapped to allowed tools, enforcing the principle of least privilege.

Role Assignment:
- LDAP lookup - if user is in Rover group "devlakemcpadmin" -> mcp-admin
- Otherwise -> mcp-viewer (all tools EXCEPT execute_query)

Configure via environment variables (set in OCP ConfigMap):
  - LDAP_ENABLED: Enable LDAP lookups (default: true if ldap3 available)
  - LDAP_ADMIN_GROUP: Rover group name for admin access (default: devlakemcpadmin)
  - RBAC_DEFAULT_ROLE: Role for non-admin users (default: mcp-viewer)
"""

import os
from typing import Any, Dict, List, Optional, Set

from utils.logger import get_logger
from utils.ldap_service import LDAPService

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


def extract_username_from_email(email: str) -> str:
    """Extract the username part from an email address."""
    if "@" in email:
        return email.split("@")[0].lower()
    return email.lower()


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
    based on LDAP Rover group membership.

    Role Resolution:
    - LDAP lookup - if user is in Rover group "devlakemcpadmin" -> mcp-admin
    - Otherwise -> mcp-viewer (no execute_query)

    Example:
        User: "daturece@redhat.com"
        LDAP check: is daturece in cn=devlakemcpadmin?
        -> Yes -> mcp-admin
        -> No -> mcp-viewer
    """

    def __init__(
        self,
        role_permissions: Optional[Dict[str, Set[str]]] = None,
        default_role: Any = _NOT_SET,
        ldap_service: Optional[LDAPService] = None,
    ):
        """
        Initialize the authorization service.

        Args:
            role_permissions: Custom role-to-permissions mapping (uses default if None)
            default_role: Default role for users without explicit group assignment.
                          Pass None to disable default roles, or omit to use env var.
            ldap_service: Optional LDAPService instance for Rover group lookups
        """
        self.logger = get_logger(f"{__name__}.AuthorizationService")
        self.role_permissions = role_permissions or ROLE_PERMISSIONS

        # Handle default_role: _NOT_SET means read from env, None means no default
        if default_role is _NOT_SET:
            self.default_role = get_default_role_from_env()
        else:
            self.default_role = default_role

        # Initialize LDAP service for Rover group lookups
        self.ldap_service = ldap_service or LDAPService()

        self.logger.info(
            f"Authorization service initialized with {len(self.role_permissions)} roles, "
            f"LDAP enabled={self.ldap_service.enabled}, "
            f"default_role={self.default_role}"
        )

    def resolve_user_roles(
        self,
        user_groups: List[str],
        user_email: Optional[str] = None,
        username: Optional[str] = None,
    ) -> List[str]:
        """
        Resolve the effective roles for a user based on LDAP Rover groups.

        Role assignment:
        - LDAP lookup - if user is in Rover group "devlakemcpadmin" -> mcp-admin
        - Otherwise -> mcp-viewer (default)

        Example:
            Username: "daturece"
            LDAP check: is daturece in cn=devlakemcpadmin?
            -> Yes -> mcp-admin
            -> No -> mcp-viewer

        Args:
            user_groups: Groups from the user's OIDC token (not used, kept for interface)
            user_email: User's email address (fallback if username not provided)
            username: User's username from OIDC token (preferred)

        Returns:
            List of resolved role names
        """
        # Prefer username from token, fallback to extracting from email
        if not username and user_email:
            username = extract_username_from_email(user_email)

        # Check LDAP Rover group membership
        if username and self.ldap_service.enabled:
            try:
                if self.ldap_service.is_admin(username):
                    self.logger.info(
                        f"Admin role assigned via LDAP Rover group "
                        f"'{self.ldap_service.admin_group}': {username}"
                    )
                    return ["mcp-admin"]
            except Exception as e:
                self.logger.warning(f"LDAP lookup failed for '{username}': {e}")

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
        username: Optional[str] = None,
    ) -> bool:
        """
        Check if a user is authorized to call a specific tool.

        Args:
            user_groups: List of groups/roles from the user's OIDC token
            tool_name: Name of the tool being called
            user_email: Optional email (fallback if username not provided)
            username: Optional username from OIDC token (preferred)

        Returns:
            True if the user is authorized, False otherwise
        """
        # Resolve effective roles (considering LDAP groups)
        effective_roles = self.resolve_user_roles(user_groups, user_email, username)

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
        self,
        user_groups: List[str],
        user_email: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Set[str]:
        """
        Get all tools a user is allowed to call based on their roles.

        Args:
            user_groups: List of groups/roles from the user's OIDC token
            user_email: Optional email (fallback if username not provided)
            username: Optional username from OIDC token (preferred)

        Returns:
            Set of allowed tool names (or {"*"} for full access)
        """
        # Resolve effective roles
        effective_roles = self.resolve_user_roles(user_groups, user_email, username)

        allowed = set()

        # Add tools from each resolved role
        for role in effective_roles:
            role_tools = self.role_permissions.get(role, set())
            if "*" in role_tools:
                return {"*"}  # Full access
            allowed.update(role_tools)

        return allowed

    def get_denied_reason(
        self,
        user_groups: List[str],
        tool_name: str,
        user_email: Optional[str] = None,
        username: Optional[str] = None,
    ) -> str:
        """
        Get a human-readable reason why access was denied.

        Args:
            user_groups: User's groups
            tool_name: Tool that was denied
            user_email: User's email (fallback if username not provided)
            username: User's username from OIDC token (preferred)

        Returns:
            Explanation string for the denial
        """
        # Resolve effective roles to show what they actually have
        effective_roles = self.resolve_user_roles(user_groups, user_email, username)

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
            "ldap": self.ldap_service.get_cache_stats() if self.ldap_service else None,
        }
