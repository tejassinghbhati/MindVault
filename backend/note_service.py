from sqlalchemy.orm import Session
import uuid
from . import models, ai_service

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    # Simple chunking logic
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def index_note(note: models.Note, db: Session):
    chunks = chunk_text(note.content)
    
    # Delete old chunks if any
    old_chunks = db.query(models.NoteChunk).filter(models.NoteChunk.note_id == note.id).all()
    if old_chunks:
        chunk_ids = [c.id for c in old_chunks]
        ai_service.ai_service.delete_from_vector_store(chunk_ids)
        for c in old_chunks:
            db.delete(c)
        db.commit()
    
    for content in chunks:
        chunk_id = str(uuid.uuid4())
        db_chunk = models.NoteChunk(
            id=chunk_id,
            note_id=note.id,
            content=content,
            embedding_id=chunk_id
        )
        db.add(db_chunk)
        
        # Add to ChromaDB
        ai_service.ai_service.add_to_vector_store(
            chunk_id=chunk_id,
            text=content,
            metadata={"note_id": note.id, "title": note.title}
        )
        
    db.commit()
