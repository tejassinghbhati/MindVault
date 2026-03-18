from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import os
import uuid
from typing import List

from . import models, database, ai_service, note_service
from .database import engine, get_db

# Pydantic Schemas
class NoteCreate(BaseModel):
    title: str
    content: str

class QueryRequest(BaseModel):
    query: str

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="MindVault API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to MindVault API"}

@app.post("/notes/", response_model=None)
def create_note(note: NoteCreate, db: Session = Depends(get_db)):
    note_id = str(uuid.uuid4())
    # Save to disk as well
    filename = f"{note_id}.md"
    storage_path = os.getenv("STORAGE_PATH", "../notes")
    os.makedirs(storage_path, exist_ok=True)
    
    with open(os.path.join(storage_path, filename), "w", encoding="utf-8") as f:
        f.write(note.content)
        
    db_note = models.Note(id=note_id, title=note.title, content=note.content, filename=filename)
    db.add(db_note)
    db.commit()
    db.refresh(db_note)
    
    # Chunk and Embed
    note_service.index_note(db_note, db)
    
    return db_note

@app.get("/notes/")
def list_notes(db: Session = Depends(get_db)):
    return db.query(models.Note).all()

@app.put("/notes/{note_id}")
def update_note(note_id: str, note: NoteCreate, db: Session = Depends(get_db)):
    db_note = db.query(models.Note).filter(models.Note.id == note_id).first()
    if not db_note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    db_note.title = note.title
    db_note.content = note.content
    db.commit()
    db.refresh(db_note)
    
    # Update file on disk
    storage_path = os.getenv("STORAGE_PATH", "../notes")
    with open(os.path.join(storage_path, db_note.filename), "w", encoding="utf-8") as f:
        f.write(note.content)
        
    # Re-index
    note_service.index_note(db_note, db)
    
    return db_note

@app.get("/notes/{note_id}")
def get_note(note_id: str, db: Session = Depends(get_db)):
    note = db.query(models.Note).filter(models.Note.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note

@app.post("/query/")
def query_notes(request: QueryRequest, db: Session = Depends(get_db)):
    # 1. Search vector store
    results = ai_service.ai_service.query_vector_store(request.query)
    
    # 2. Extract context
    context = "\n---\n".join(results['documents'][0])
    
    # 3. Generate answer
    answer = ai_service.ai_service.generate_answer(request.query, context)
    
    return {
        "answer": answer,
        "context": results['documents'][0],
        "sources": results['metadatas'][0]
    }

@app.get("/related/{note_id}")
def get_related_notes(note_id: str, db: Session = Depends(get_db)):
    note = db.query(models.Note).filter(models.Note.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    # Search vector store using the note's content as query
    results = ai_service.ai_service.query_vector_store(note.content, n_results=5)
    
    # Filter out current note chunks
    related = []
    seen_notes = {note_id}
    for metadata in results['metadatas'][0]:
        r_note_id = metadata['note_id']
        if r_note_id not in seen_notes:
            related_note = db.query(models.Note).filter(models.Note.id == r_note_id).first()
            if related_note:
                related.append(related_note)
                seen_notes.add(r_note_id)
                
    return related
