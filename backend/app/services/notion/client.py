import re
from typing import Any, Optional
import httpx

from app.core.config import settings
from app.core.errors import (
    NotionAPIError,
    NotionAuthError,
    NotionPageNotFoundError,
    NotionRateLimitError,
)


class NotionClient:
    """Async HTTP client interacting with the official Notion REST API (v1).
    Guarantees strict authentication validation, rate limiting resilience,
    pagination handling, and zero credential leakage.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.token = token or settings.NOTION_API_KEY
        self.base_url = (base_url or settings.NOTION_API_BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.NOTION_TIMEOUT_SECONDS
        self.notion_version = settings.NOTION_API_VERSION

    def _get_headers(self) -> dict[str, str]:
        if not self.token:
            raise NotionAuthError("No Notion API token configured or provided.")
        return {
            "Authorization": f"Bearer {self.token.strip()}",
            "Notion-Version": self.notion_version,
            "Content-Type": "application/json",
        }

    def _handle_response_errors(self, res: httpx.Response, entity_id: str = "") -> None:
        if res.status_code == 401 or res.status_code == 403:
            raise NotionAuthError(f"Notion authentication failed (HTTP {res.status_code}): Token is invalid or access denied.")
        elif res.status_code == 404:
            raise NotionPageNotFoundError(entity_id or "unknown")
        elif res.status_code == 429:
            retry_after_hdr = res.headers.get("Retry-After")
            retry_after = float(retry_after_hdr) if retry_after_hdr and retry_after_hdr.isdigit() else None
            raise NotionRateLimitError(retry_after=retry_after)
        elif res.status_code >= 400:
            error_data = {}
            try:
                error_data = res.json()
            except Exception:
                pass
            msg = error_data.get("message", res.text[:200])
            raise NotionAPIError(f"HTTP {res.status_code}: {msg}", status_code=res.status_code)

    async def get_bot_user(self) -> dict[str, Any]:
        """Queries /v1/users/me to verify credentials and obtain bot/workspace metadata."""
        headers = self._get_headers()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.get(f"{self.base_url}/users/me", headers=headers)
                if res.status_code != 200:
                    self._handle_response_errors(res)
                return res.json()
        except httpx.TimeoutException:
            raise NotionAPIError(f"Notion request timed out after {self.timeout}s.", status_code=504)
        except httpx.ConnectError:
            raise NotionAPIError("Could not establish connection to Notion API server.", status_code=503)
        except (NotionAuthError, NotionRateLimitError, NotionPageNotFoundError, NotionAPIError):
            raise
        except Exception as e:
            raise NotionAPIError(str(e))

    async def search_pages(
        self,
        query: str = "",
        start_cursor: Optional[str] = None,
        page_size: int = 100,
    ) -> dict[str, Any]:
        """Searches for accessible pages within the authorized Notion integration."""
        headers = self._get_headers()
        payload: dict[str, Any] = {
            "query": query,
            "filter": {"value": "page", "property": "object"},
            "page_size": min(page_size, 100),
        }
        if start_cursor:
            payload["start_cursor"] = start_cursor

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(f"{self.base_url}/search", headers=headers, json=payload)
                if res.status_code != 200:
                    self._handle_response_errors(res)
                return res.json()
        except httpx.TimeoutException:
            raise NotionAPIError(f"Notion search request timed out after {self.timeout}s.", status_code=504)
        except httpx.ConnectError:
            raise NotionAPIError("Could not establish connection to Notion API server.", status_code=503)
        except (NotionAuthError, NotionRateLimitError, NotionPageNotFoundError, NotionAPIError):
            raise
        except Exception as e:
            raise NotionAPIError(str(e))

    async def fetch_all_pages(self, query: str = "") -> list[dict[str, Any]]:
        """Handles pagination to retrieve all accessible pages."""
        all_pages: list[dict[str, Any]] = []
        has_more = True
        next_cursor = None

        while has_more:
            data = await self.search_pages(query=query, start_cursor=next_cursor, page_size=100)
            results = data.get("results", [])
            for item in results:
                if item.get("object") == "page":
                    all_pages.append(item)
            has_more = bool(data.get("has_more"))
            next_cursor = data.get("next_cursor")
            if not next_cursor:
                break

        return all_pages

    async def get_page(self, page_id: str) -> dict[str, Any]:
        """Fetches page metadata and properties for a specific page ID."""
        clean_id = page_id.replace("-", "")
        headers = self._get_headers()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.get(f"{self.base_url}/pages/{clean_id}", headers=headers)
                if res.status_code != 200:
                    self._handle_response_errors(res, entity_id=page_id)
                return res.json()
        except httpx.TimeoutException:
            raise NotionAPIError(f"Notion request for page '{page_id}' timed out.", status_code=504)
        except httpx.ConnectError:
            raise NotionAPIError("Could not establish connection to Notion API server.", status_code=503)
        except (NotionAuthError, NotionRateLimitError, NotionPageNotFoundError, NotionAPIError):
            raise
        except Exception as e:
            raise NotionAPIError(str(e))

    async def get_block_children(
        self,
        block_id: str,
        start_cursor: Optional[str] = None,
        page_size: int = 100,
    ) -> dict[str, Any]:
        """Fetches child blocks for a given block or page ID."""
        clean_id = block_id.replace("-", "")
        headers = self._get_headers()
        params: dict[str, Any] = {"page_size": min(page_size, 100)}
        if start_cursor:
            params["start_cursor"] = start_cursor

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.get(
                    f"{self.base_url}/blocks/{clean_id}/children",
                    headers=headers,
                    params=params,
                )
                if res.status_code != 200:
                    self._handle_response_errors(res, entity_id=block_id)
                return res.json()
        except httpx.TimeoutException:
            raise NotionAPIError(f"Notion blocks request timed out for '{block_id}'.", status_code=504)
        except httpx.ConnectError:
            raise NotionAPIError("Could not establish connection to Notion API server.", status_code=503)
        except (NotionAuthError, NotionRateLimitError, NotionPageNotFoundError, NotionAPIError):
            raise
        except Exception as e:
            raise NotionAPIError(str(e))

    async def fetch_all_blocks(self, block_id: str, max_depth: int = 4) -> list[dict[str, Any]]:
        """Recursively retrieves all child blocks for a page, handling pagination and nested blocks."""
        all_blocks: list[dict[str, Any]] = []
        has_more = True
        next_cursor = None

        while has_more:
            data = await self.get_block_children(block_id, start_cursor=next_cursor, page_size=100)
            results = data.get("results", [])
            for block in results:
                # If block has children (e.g., table with table_row, toggle, or nested list)
                if block.get("has_children") and max_depth > 0:
                    child_id = block.get("id")
                    if child_id:
                        try:
                            children = await self.fetch_all_blocks(child_id, max_depth=max_depth - 1)
                            block["child_blocks"] = children
                        except Exception:
                            block["child_blocks"] = []
                all_blocks.append(block)

            has_more = bool(data.get("has_more"))
            next_cursor = data.get("next_cursor")
            if not next_cursor:
                break

        return all_blocks
