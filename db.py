from sqlmodel import SQLModel, create_engine

engine = create_engine(
    "sqlite:///clinic_assist.db",
    echo=False,
    connect_args={"check_same_thread": False},
)

def init_db() -> None:
    SQLModel.metadata.create_all(engine)
