class AppException(Exception):
    """Base exception for application errors."""

    def __init__(self, message: str, status_code: int = 500, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class UnsupportedFileTypeError(AppException):
    def __init__(self, extension: str, allowed: set[str]):
        super().__init__(
            message=f"Unsupported file type '.{extension}'. Allowed types: {', '.join(sorted(allowed))}",
            status_code=415,
            details={"extension": extension, "allowed": list(allowed)},
        )


class FileSizeExceededError(AppException):
    def __init__(self, actual_size_bytes: int, max_size_bytes: int):
        super().__init__(
            message=f"File size ({actual_size_bytes} bytes) exceeds maximum permitted limit ({max_size_bytes} bytes).",
            status_code=413,
            details={"actual_size": actual_size_bytes, "max_size": max_size_bytes},
        )


class EmptyDocumentError(AppException):
    def __init__(self, filename: str):
        super().__init__(
            message=f"Document '{filename}' contains no readable text content.",
            status_code=422,
            details={"filename": filename},
        )


class DocumentParsingError(AppException):
    def __init__(self, filename: str, reason: str):
        super().__init__(
            message=f"Failed to parse document '{filename}': {reason}",
            status_code=422,
            details={"filename": filename, "reason": reason},
        )


class UnsupportedEncodingError(AppException):
    def __init__(self, filename: str, detected_encoding: str | None = None):
        super().__init__(
            message=f"Document '{filename}' has unsupported text encoding (detected: {detected_encoding or 'unknown'}). Please provide UTF-8 encoded text.",
            status_code=422,
            details={"filename": filename, "detected_encoding": detected_encoding},
        )


class DocumentNotFoundError(AppException):
    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document with ID '{document_id}' was not found.",
            status_code=404,
            details={"document_id": document_id},
        )


class DuplicateDocumentError(AppException):
    def __init__(self, filename: str, existing_doc_id: str, sha256_hash: str):
        super().__init__(
            message=f"Document '{filename}' is identical to existing document '{existing_doc_id}' in this workspace.",
            status_code=409,
            details={"filename": filename, "existing_doc_id": existing_doc_id, "sha256": sha256_hash},
        )


class OllamaUnavailableError(AppException):
    def __init__(self, base_url: str):
        super().__init__(
            message=f"Local Ollama service is unreachable at {base_url}. Ensure Ollama is running (`ollama serve`).",
            status_code=503,
            details={"base_url": base_url, "instructions": "Run `ollama serve` to start the local Ollama instance."},
        )


class OllamaModelNotFoundError(AppException):
    def __init__(self, model_name: str, available_models: list[str]):
        super().__init__(
            message=f"Configured model '{model_name}' is not installed in local Ollama. Run `ollama pull {model_name}` to download it.",
            status_code=503,
            details={
                "missing_model": model_name,
                "available_models": available_models,
                "setup_instruction": f"Run `ollama pull {model_name}` in your terminal.",
            },
        )


class OllamaTimeoutError(AppException):
    def __init__(self, model_name: str, timeout_seconds: float):
        super().__init__(
            message=f"Ollama inference timed out after {timeout_seconds} seconds using model '{model_name}'. If the model was cold-loading into RAM, please retry your question now.",
            status_code=504,
            details={"model": model_name, "timeout_seconds": timeout_seconds},
        )


class OllamaGenerationError(AppException):
    def __init__(self, reason: str):
        super().__init__(
            message=f"Ollama generation failed: {reason}",
            status_code=502,
            details={"reason": reason},
        )


class AuthenticationError(AppException):
    def __init__(self, message: str = "Authentication failed.", details: dict | None = None):
        super().__init__(message=message, status_code=401, details=details or {})


class InvalidCredentialsError(AuthenticationError):
    def __init__(self):
        super().__init__(message="Invalid username or password.")


class InvalidTokenError(AuthenticationError):
    def __init__(self, reason: str = "Invalid or malformed authentication token."):
        super().__init__(message=reason)


class TokenExpiredError(AuthenticationError):
    def __init__(self):
        super().__init__(message="Authentication token has expired. Please log in again.")


class PermissionDeniedError(AppException):
    def __init__(self, message: str = "You do not have permission to perform this action.", details: dict | None = None):
        super().__init__(message=message, status_code=403, details=details or {})


class WorkspaceAccessDeniedError(PermissionDeniedError):
    def __init__(self, workspace_id: str):
        super().__init__(
            message=f"Access denied: You are not a member of workspace '{workspace_id}'.",
            details={"workspace_id": workspace_id},
        )


class DocumentAccessDeniedError(PermissionDeniedError):
    def __init__(self, document_id: str, action: str = "read"):
        super().__init__(
            message=f"Access denied: You do not have '{action}' permission for document '{document_id}'.",
            details={"document_id": document_id, "action": action},
        )


class UserAlreadyExistsError(AppException):
    def __init__(self, identifier: str):
        super().__init__(
            message=f"A user with identifier '{identifier}' already exists.",
            status_code=409,
            details={"identifier": identifier},
        )


class UserNotFoundError(AppException):
    def __init__(self, user_id: str):
        super().__init__(
            message=f"User with ID '{user_id}' was not found.",
            status_code=404,
            details={"user_id": user_id},
        )


class NotionError(AppException):
    """Base exception for Notion integration errors."""
    def __init__(self, message: str, status_code: int = 502, details: dict | None = None):
        super().__init__(message=message, status_code=status_code, details=details or {})


class NotionAuthError(NotionError):
    def __init__(self, message: str = "Invalid, missing, or revoked Notion API credentials."):
        super().__init__(message=message, status_code=401)


class NotionRateLimitError(NotionError):
    def __init__(self, retry_after: float | None = None):
        msg = f"Notion API rate limit exceeded. Retry after {retry_after}s." if retry_after else "Notion API rate limit exceeded."
        super().__init__(message=msg, status_code=429, details={"retry_after": retry_after})


class NotionPageNotFoundError(NotionError):
    def __init__(self, page_id: str):
        super().__init__(message=f"Notion page '{page_id}' was not found or access was revoked.", status_code=404, details={"page_id": page_id})


class NotionAPIError(NotionError):
    def __init__(self, reason: str, status_code: int = 502):
        super().__init__(message=f"Notion API communication failed: {reason}", status_code=status_code, details={"reason": reason})

