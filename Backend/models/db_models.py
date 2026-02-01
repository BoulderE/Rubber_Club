from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

# 表 1: 用戶表
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    pin = Column(String(10), unique=True, nullable=False)
    name = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 關聯到運動記錄
    exercise_records = relationship("ExerciseRecord", back_populates="user")

# 表 2: 運動記錄表
class ExerciseRecord(Base):
    __tablename__ = 'exercise_records'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    exercise_name = Column(String(100), nullable=False)
    accuracy = Column(Float, nullable=True)
    smoothness = Column(Float, nullable=True)
    duration = Column(Float, nullable=True)  # 秒
    rep_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="exercise_records")

# 表 3: 動作規則表
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


# 數據庫連接
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