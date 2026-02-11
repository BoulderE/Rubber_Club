from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Text, DateTime, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    pin = Column(String(10), unique=True, nullable=False)
    name = Column(String(50), nullable=True)
    role = Column(String(20), default='user')  
    created_at = Column(DateTime, default=datetime.utcnow)
    
    exercise_records = relationship("ExerciseRecord", back_populates="user")

    assigned_exercises = relationship("AssignedExercise", back_populates="user")

class ExerciseRecord(Base):
    __tablename__ = 'exercise_records'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    exercise_name = Column(String(100), nullable=False)
    accuracy = Column(Float, nullable=True)
    smoothness = Column(Float, nullable=True)
    duration = Column(Float, nullable=True)  
    rep_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="exercise_records")


class ExerciseRule(Base):
    __tablename__ = 'exercise_rules'
    
    id = Column(Integer, primary_key=True)
    exercise_key = Column(String(50), unique=True, nullable=False)  
    name = Column(String(100), nullable=False)  
    landmarks_to_use = Column(JSON, nullable=False)  
    logic_function = Column(String(100), nullable=False)  
    params = Column(JSON, nullable=False)  
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AssignedExercise(Base):
    __tablename__ = 'assigned_exercises'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    exercise_key = Column(String(50), nullable=False)  
    exercise_name = Column(String(100), nullable=False)
    
    target_reps = Column(Integer, default=10)  
    target_sets = Column(Integer, default=3)   
    difficulty = Column(String(20), default='beginner')  
    
    completed_sets = Column(Integer, default=0)
    completed_reps_total = Column(Integer, default=0)  
    avg_smoothness = Column(Float, default=0.0)  
    
    status = Column(String(20), default='pending')
    
    assigned_date = Column(Date, default=datetime.utcnow)
    due_date = Column(Date, nullable=True)  
    completed_at = Column(DateTime, nullable=True)  
    
    admin_notes = Column(Text, nullable=True)  
    user_notes = Column(Text, nullable=True)   
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="assigned_exercises")


def get_engine():
    return create_engine(os.getenv('DATABASE_URL'))

def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("數據庫表創建成功！")

def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()