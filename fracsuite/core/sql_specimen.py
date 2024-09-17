from typing import List, Optional
from sqlalchemy import ForeignKey, String, Float, Integer, Boolean, PickleType, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
import numpy as np
from tqdm import tqdm

from fracsuite import acc
from fracsuite.core.accelerationdata import AccelerationData
from fracsuite.core.anisotropy_images import AnisotropyImages
from fracsuite.core.specimen import Specimen

class Base(DeclarativeBase):
    pass

class SSpecimen(Base):
    __tablename__ = 'specimens'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    thickness: Mapped[int] = mapped_column(Integer, default=0)
    nom_stress: Mapped[int] = mapped_column(Integer, default=0)
    boundary: Mapped[str] = mapped_column(String, default="Unknown")
    nbr: Mapped[int] = mapped_column(Integer, default=0)
    break_mode: Mapped[str] = mapped_column(String, default="punch")
    break_pos: Mapped[str] = mapped_column(String, default="corner")
    
    fall_height_m: Mapped[float] = mapped_column(Float, default=0.07)
    fall_repeat: Mapped[int] = mapped_column(Integer, default=1)
    real_size_mm: Mapped[tuple[int, int]] = mapped_column(PickleType, default=(500, 500))
    
    path: Mapped[str] = mapped_column(String, nullable=False)
    excluded_sensor_positions: Mapped[List[int]] = mapped_column(PickleType, default=[])
    exclude_all_sensors: Mapped[bool] = mapped_column(Boolean, default=False)
    excluded_positions: Mapped[List[int]] = mapped_column(PickleType, default=[])
    excluded_positions_radius: Mapped[float] = mapped_column(Float, default=100.0)
    actual_break_pos: Mapped[Optional[tuple[float, float]]] = mapped_column(PickleType, default=None)
    custom_break_pos_exclusion_radius: Mapped[float] = mapped_column(Float, default=20.0)
    custom_edge_exclusion_distance: Mapped[float] = mapped_column(Float, default=10.0)
    acc_data: Mapped[List[np.ndarray]] = mapped_column(PickleType, default="")
        
    fracture_image: Mapped[Optional[np.ndarray]] = mapped_column(PickleType, default=None, nullable=True)
    anisotropy_scans: Mapped[Optional[List[np.ndarray]]] = mapped_column(PickleType, default=None, nullable=True)

    def __repr__(self) -> str:
        return (f"SSpecimen(id={self.id!r})")

# Create a PostgreSQL database connection
DATABASE_URL = "postgresql://postgres:examplepassword@localhost:5432/fractures"
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Example usage
if __name__ == "__main__":
    # Create a new specimen        
    
    # for s in tqdm(Specimen.get_all()):
    #     s: Specimen
    #     newspecimen = SSpecimen(name=s.name, path=s.path, fall_height_m=s.fall_height_m, real_size_mm=s.get_real_size(), fall_repeat=s.get_setting(s.SET_FALLREPEAT), 
    #                             excluded_sensor_positions=s.get_setting(s.SET_EXCLUDED_SENSOR_POSITIONS), 
    #                             exclude_all_sensors=s.get_setting(s.SET_EXCLUDE_ALL_SENSORS), excluded_positions=s.get_setting(s.SET_EXCLUDED_POSITIONS), 
    #                             excluded_positions_radius=s.get_setting(s.SET_EXCLUDED_POSITIONS_RADIUS), 
    #                             break_mode=s.break_mode, break_pos=s.break_pos, 
    #                             actual_break_pos=s.get_setting(s.SET_BREAKPOS), 
    #                             custom_break_pos_exclusion_radius=s.get_setting(s.SET_CBREAKPOSEXCL), 
    #                             custom_edge_exclusion_distance=s.get_setting(s.SET_EDGEEXCL), 
    #                             acc_data = None,
    #                             boundary=s.boundary, nom_stress=s.nom_stress, thickness=s.thickness, 
    #                             fracture_image=s.get_fracture_image(), 
    #                             anisotropy_scans=s.anisotropy)
    
    #     session.add(newspecimen)
    #     session.commit()

    # Query the specimen
    specimen = session.query(SSpecimen).limit(50).all()
    print(specimen)