from pydantic import BaseModel, Field
from typing import Literal

# Este nuevo esquema es altamente intuitivo para realizar pruebas
class PatientData(BaseModel):
    age: float = Field(..., description="Edad del paciente")
    gender: Literal['Male', 'Female', 'Other'] = Field('Male', description="Género")
    hypertension: int = Field(0, description="1 si tiene hipertensión, 0 si no")
    heart_disease: int = Field(0, description="1 si tiene enfermedad cardíaca, 0 si no")
    ever_married: Literal['Yes', 'No'] = Field('Yes', description="¿Se ha casado alguna vez?")
    work_type: Literal['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'] = Field('Private', description="Tipo de trabajo principal")
    Residence_type: Literal['Urban', 'Rural'] = Field('Urban', description="Tipo de zona residencial")
    avg_glucose_level: float = Field(..., description="Nivel promedio de glucosa en sangre")
    bmi: float = Field(..., description="Índice de masa corporal")
    smoking_status: Literal['formerly smoked', 'never smoked', 'smokes', 'Unknown'] = Field('never smoked', description="Estatus de tabaquismo")
