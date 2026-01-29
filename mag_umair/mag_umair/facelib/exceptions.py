# FaceLibrary Exceptions


class FaceLibraryError(Exception):
    """Base exception for FaceLibrary operations."""
    pass


class NoFaceDetectedError(FaceLibraryError):
    """Raised when no face is found in the image."""
    
    def __init__(self, message: str = "No face detected in image"):
        super().__init__(message)


class FilterRejectedError(FaceLibraryError):
    """Raised when all faces in an image are rejected by filters."""
    
    def __init__(self, message: str = "All faces rejected by quality filters", reasons: list = None):
        super().__init__(message)
        self.reasons = reasons or []


class FaceNotFoundError(FaceLibraryError):
    """Raised when a face_id doesn't exist in the index."""
    
    def __init__(self, face_id: str):
        super().__init__(f"Face not found: {face_id}")
        self.face_id = face_id


class IndexNotLoadedError(FaceLibraryError):
    """Raised when an operation requires a loaded index but none exists."""
    
    def __init__(self, message: str = "No index loaded. Call load() or index some images first."):
        super().__init__(message)


class ImageLoadError(FaceLibraryError):
    """Raised when an image cannot be loaded."""
    
    def __init__(self, path: str, reason: str = None):
        message = f"Failed to load image: {path}"
        if reason:
            message += f" ({reason})"
        super().__init__(message)
        self.path = path
