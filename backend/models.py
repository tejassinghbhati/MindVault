from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class Note(Base):
    __tablename__ = "notes"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, index=True)
    filename = Column(String, unique=True)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    chunks = relationship("NoteChunk", back_populates="note", cascade="all, delete-orphan")

class NoteChunk(Base):
    __tablename__ = "note_chunks"

    id = Column(String, primary_key=True, index=True)
    note_id = Column(String, ForeignKey("notes.id"))
    content = Column(Text)
    embedding_id = Column(String)  # ID in ChromaDB

    note = relationship("Note", back_populates="chunks")
