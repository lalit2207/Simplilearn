"""Read uploaded files and, when they are medical records, file them on a chart."""

from src.attachments.extract import ExtractedFile, extract_upload
from src.attachments.ingest import PreparedTurn, prepare_turn

__all__ = ["ExtractedFile", "PreparedTurn", "extract_upload", "prepare_turn"]
