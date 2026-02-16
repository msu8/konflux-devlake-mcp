#!/usr/bin/env python3
"""
LDAP Service for Rover Group Lookups

This module provides LDAP-based group membership checks against Red Hat's
corporate LDAP (ldap.corp.redhat.com). It is used to determine if a user
is a member of specific Rover groups for RBAC purposes.

Features:
- Caching of LDAP results to reduce latency
- Configurable via environment variables
- Fallback behavior when LDAP is unavailable
"""

import os
import time
from typing import Dict, Optional, Set

from utils.logger import get_logger

# Try to import ldap3, but make it optional for environments without LDAP access
try:
    from ldap3 import Connection, Server, SUBTREE, ALL
    from ldap3.core.exceptions import LDAPException

    LDAP_AVAILABLE = True
except ImportError:
    LDAP_AVAILABLE = False
    LDAPException = Exception  # Fallback for type hints


# Default configuration
DEFAULT_LDAP_SERVER = "ldaps://ldap.corp.redhat.com"
DEFAULT_LDAP_BASE_DN = "dc=redhat,dc=com"
DEFAULT_CACHE_TTL = 300  # 5 minutes


class LDAPGroupCache:
    """Simple TTL cache for LDAP group membership results."""

    def __init__(self, ttl_seconds: int = DEFAULT_CACHE_TTL):
        """
        Initialize the cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self._cache: Dict[str, tuple] = {}  # username -> (groups, timestamp)
        self._ttl = ttl_seconds

    def get(self, username: str) -> Optional[Set[str]]:
        """
        Get cached groups for a user.

        Args:
            username: The username to lookup

        Returns:
            Set of group names if cached and not expired, None otherwise
        """
        if username not in self._cache:
            return None

        groups, timestamp = self._cache[username]
        if time.time() - timestamp > self._ttl:
            # Cache expired
            del self._cache[username]
            return None

        return groups

    def set(self, username: str, groups: Set[str]) -> None:
        """
        Cache groups for a user.

        Args:
            username: The username
            groups: Set of group names
        """
        self._cache[username] = (groups, time.time())

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def size(self) -> int:
        """Get the number of cached entries."""
        return len(self._cache)


class LDAPService:
    """
    Service for querying Red Hat corporate LDAP for Rover group membership.

    This service checks if users are members of specific Rover groups
    (e.g., devlakemcpadmin) by querying LDAP.

    Environment variables:
        LDAP_SERVER: LDAP server URL (default: ldaps://ldap.corp.redhat.com)
        LDAP_BASE_DN: Base DN for searches (default: dc=redhat,dc=com)
        LDAP_CACHE_TTL: Cache TTL in seconds (default: 300)
        LDAP_ENABLED: Enable/disable LDAP lookups (default: true if ldap3 available)
        LDAP_ADMIN_GROUP: Rover group name for admin access (default: devlakemcpadmin)
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        base_dn: Optional[str] = None,
        cache_ttl: Optional[int] = None,
        enabled: Optional[bool] = None,
        admin_group: Optional[str] = None,
    ):
        """
        Initialize the LDAP service.

        Args:
            server_url: LDAP server URL (or use LDAP_SERVER env var)
            base_dn: Base DN for LDAP searches (or use LDAP_BASE_DN env var)
            cache_ttl: Cache TTL in seconds (or use LDAP_CACHE_TTL env var)
            enabled: Enable LDAP lookups (or use LDAP_ENABLED env var)
            admin_group: Rover group name for admin access (or use LDAP_ADMIN_GROUP env var)
        """
        self.logger = get_logger(f"{__name__}.LDAPService")

        # Configuration from parameters or environment
        self.server_url = server_url or os.environ.get("LDAP_SERVER", DEFAULT_LDAP_SERVER)
        self.base_dn = base_dn or os.environ.get("LDAP_BASE_DN", DEFAULT_LDAP_BASE_DN)
        self.admin_group = admin_group or os.environ.get("LDAP_ADMIN_GROUP", "devlakemcpadmin")

        cache_ttl_val = cache_ttl or int(os.environ.get("LDAP_CACHE_TTL", str(DEFAULT_CACHE_TTL)))
        self._cache = LDAPGroupCache(cache_ttl_val)

        # Determine if LDAP is enabled
        if enabled is not None:
            self._enabled = enabled
        else:
            env_enabled = os.environ.get("LDAP_ENABLED", "").lower()
            if env_enabled:
                self._enabled = env_enabled == "true"
            else:
                # Default: enabled if ldap3 is available
                self._enabled = LDAP_AVAILABLE

        if self._enabled and not LDAP_AVAILABLE:
            self.logger.warning(
                "LDAP_ENABLED is true but ldap3 library is not installed. "
                "LDAP lookups will be disabled."
            )
            self._enabled = False

        if self._enabled:
            self.logger.info(
                f"LDAP service initialized: server={self.server_url}, "
                f"admin_group={self.admin_group}, cache_ttl={cache_ttl_val}s"
            )
        else:
            self.logger.info("LDAP service disabled")

    @property
    def enabled(self) -> bool:
        """Check if LDAP lookups are enabled."""
        return self._enabled

    def get_user_groups(self, username: str) -> Set[str]:
        """
        Get Rover groups for a user from LDAP.

        Args:
            username: The username to lookup (e.g., "daturece")

        Returns:
            Set of Rover group names the user is a member of
        """
        if not self._enabled:
            return set()

        # Check cache first
        cached = self._cache.get(username)
        if cached is not None:
            self.logger.debug(f"LDAP cache hit for user '{username}': {len(cached)} groups")
            return cached

        # Query LDAP
        groups = self._query_ldap_groups(username)

        # Cache the result
        self._cache.set(username, groups)

        return groups

    def _query_ldap_groups(self, username: str) -> Set[str]:
        """
        Query LDAP for a user's group memberships.

        Args:
            username: The username to lookup

        Returns:
            Set of group names extracted from memberOf attributes
        """
        if not LDAP_AVAILABLE:
            return set()

        groups = set()

        try:
            # Create LDAP connection (anonymous bind for read-only queries)
            server = Server(self.server_url, get_info=ALL)
            conn = Connection(server, auto_bind=True)

            # Search for the user
            search_filter = f"(uid={username})"
            conn.search(
                search_base=self.base_dn,
                search_filter=search_filter,
                search_scope=SUBTREE,
                attributes=["memberOf"],
            )

            if conn.entries:
                entry = conn.entries[0]
                # Extract group names from memberOf DNs
                # Format: cn=groupname,ou=adhoc,ou=managedGroups,dc=redhat,dc=com
                for member_of in entry.memberOf.values if hasattr(entry, "memberOf") else []:
                    # Extract CN (group name) from the DN
                    if member_of.startswith("cn="):
                        group_name = member_of.split(",")[0][3:]  # Remove "cn=" prefix
                        groups.add(group_name)

            conn.unbind()

            self.logger.info(f"LDAP query for '{username}': found {len(groups)} groups")
            if self.admin_group in groups:
                self.logger.info(f"User '{username}' is member of admin group '{self.admin_group}'")

        except LDAPException as e:
            self.logger.error(f"LDAP query failed for '{username}': {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during LDAP query for '{username}': {e}")

        return groups

    def is_admin(self, username: str) -> bool:
        """
        Check if a user is a member of the admin Rover group.

        Args:
            username: The username to check

        Returns:
            True if the user is in the admin group, False otherwise
        """
        if not self._enabled:
            return False

        groups = self.get_user_groups(username)
        return self.admin_group in groups

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics for monitoring.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "enabled": self._enabled,
            "server": self.server_url if self._enabled else None,
            "admin_group": self.admin_group,
            "cache_size": self._cache.size(),
            "cache_ttl": self._cache._ttl,
        }

    def clear_cache(self) -> None:
        """Clear the group membership cache."""
        self._cache.clear()
        self.logger.info("LDAP cache cleared")
