from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import date as date_type, datetime
from enum import Enum


class Gender(str, Enum):
    """Patient gender"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class ConditionStatus(str, Enum):
    """Status of a medical condition"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    REMISSION = "remission"
    RECURRENCE = "recurrence"


class Demographics(BaseModel):
    """Patient demographic information"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    patient_id: str = Field(..., description="Unique patient identifier")
    date_of_birth: date_type = Field(..., description="Patient's date of birth")
    age: int = Field(..., description="Current age in years")
    gender: Gender
    race: Optional[str] = Field(None, description="Patient's race/ethnicity")
    marital_status: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None


class Condition(BaseModel):
    """Medical condition/diagnosis"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    code: str = Field(..., description="SNOMED or ICD code")
    display_name: str = Field(..., description="Human-readable condition name")
    onset_date: Optional[date_type] = Field(None, description="When condition started")
    abatement_date: Optional[date_type] = Field(None, description="When condition ended/resolved")
    status: ConditionStatus = Field(default=ConditionStatus.ACTIVE)
    severity: Optional[str] = Field(None, description="mild, moderate, severe")
    body_site: Optional[str] = Field(None, description="Location in body if applicable")
    
    def duration_days(self) -> Optional[int]:
        """Calculate how long the condition has been present"""
        if self.onset_date:
            end_date = self.abatement_date or date_type.today()
            return (end_date - self.onset_date).days
        return None


class Medication(BaseModel):
    """Medication information"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    code: Optional[str] = Field(None, description="RxNorm code")
    display_name: str = Field(..., description="Medication name")
    dosage: Optional[str] = Field(None, description="Dosage information")
    route: Optional[str] = Field(None, description="Route of administration (oral, IV, etc)")
    start_date: Optional[date_type] = Field(None, description="When medication started")
    end_date: Optional[date_type] = Field(None, description="When medication ended")
    status: str = Field(default="active", description="active, stopped, completed")
    reason: Optional[str] = Field(None, description="Reason for medication")


class Observation(BaseModel):
    """Lab result or vital sign observation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    code: str = Field(..., description="LOINC code")
    display_name: str = Field(..., description="Name of observation")
    value: Optional[float] = Field(None, description="Numeric value")
    value_string: Optional[str] = Field(None, description="String value for non-numeric results")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    reference_range_low: Optional[float] = None
    reference_range_high: Optional[float] = None
    date: date_type = Field(..., description="Date of observation")
    category: Optional[str] = Field(None, description="vital-signs, laboratory, etc")
    
    def is_abnormal(self) -> Optional[bool]:
        """Check if value is outside reference range"""
        if self.value and self.reference_range_low and self.reference_range_high:
            return self.value < self.reference_range_low or self.value > self.reference_range_high
        return None


class Procedure(BaseModel):
    """Medical procedure or surgery"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    code: str = Field(..., description="SNOMED or CPT code")
    display_name: str = Field(..., description="Procedure name")
    performed_date: date_type = Field(..., description="When procedure was performed")
    body_site: Optional[str] = Field(None, description="Body site where performed")
    reason: Optional[str] = Field(None, description="Reason for procedure")
    outcome: Optional[str] = Field(None, description="Outcome of procedure")


class Allergy(BaseModel):
    """Allergy or intolerance"""
    substance: str = Field(..., description="What the patient is allergic to")
    criticality: Optional[str] = Field(None, description="low, high, unable-to-assess")
    reaction: Optional[List[str]] = Field(default_factory=list, description="Reactions experienced")
    onset_date: Optional[date_type] = None
    type: str = Field(default="allergy", description="allergy or intolerance")


class LifestyleFactors(BaseModel):
    """Lifestyle and social history"""
    smoking_status: Optional[str] = Field(None, description="Current smoking status")
    alcohol_use: Optional[str] = Field(None, description="Alcohol consumption level")
    exercise_frequency: Optional[str] = None
    occupation: Optional[str] = None


class Patient(BaseModel):
    """Complete patient record for clinical trial matching"""
    demographics: Demographics
    conditions: List[Condition] = Field(default_factory=list)
    medications: List[Medication] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    procedures: List[Procedure] = Field(default_factory=list)
    allergies: List[Allergy] = Field(default_factory=list)
    lifestyle: Optional[LifestyleFactors] = None
    
    def get_active_conditions(self) -> List[Condition]:
        """Get only currently active conditions"""
        return [c for c in self.conditions if c.status == ConditionStatus.ACTIVE]
    
    def get_current_medications(self) -> List[Medication]:
        """Get medications currently being taken"""
        return [m for m in self.medications if m.status == "active"]
    
    def get_recent_observations(self, days: int = 90) -> List[Observation]:
        """Get observations from the last N days"""
        cutoff = date_type.today()
        from datetime import timedelta
        cutoff = cutoff - timedelta(days=days)
        return [o for o in self.observations if o.date >= cutoff]
    
    def to_text_summary(self) -> str:
        """
        Generate a text summary for RAG system embedding.
        This creates a human-readable summary that can be embedded and searched.
        """
        summary_parts = []
        
        # Demographics
        d = self.demographics
        summary_parts.append(
            f"Patient is a {d.age}-year-old {d.gender.value}"
        )
        if d.race:
            summary_parts[-1] += f" {d.race}"
        summary_parts[-1] += "."
        
        # Active conditions
        active_conditions = self.get_active_conditions()
        if active_conditions:
            conditions_str = ", ".join([c.display_name for c in active_conditions])
            summary_parts.append(f"Active diagnoses: {conditions_str}.")
        
        # Current medications
        current_meds = self.get_current_medications()
        if current_meds:
            meds_str = ", ".join([m.display_name for m in current_meds])
            summary_parts.append(f"Current medications: {meds_str}.")
        
        # Recent key observations (you can filter for specific important ones)
        recent_obs = self.get_recent_observations(days=90)
        if recent_obs:
            obs_str = "; ".join([
                f"{o.display_name}: {o.value} {o.unit}" if o.value 
                else f"{o.display_name}: {o.value_string}"
                for o in recent_obs[:5]  # Limit to 5 most recent
            ])
            summary_parts.append(f"Recent observations: {obs_str}.")
        
        # Procedures
        if self.procedures:
            recent_procedures = sorted(self.procedures, key=lambda p: p.performed_date, reverse=True)[:3]
            proc_str = ", ".join([p.display_name for p in recent_procedures])
            summary_parts.append(f"Recent procedures: {proc_str}.")
        
        # Allergies
        if self.allergies:
            allergy_str = ", ".join([a.substance for a in self.allergies])
            summary_parts.append(f"Known allergies: {allergy_str}.")
        
        # Lifestyle
        if self.lifestyle:
            if self.lifestyle.smoking_status:
                summary_parts.append(f"Smoking status: {self.lifestyle.smoking_status}.")
            if self.lifestyle.alcohol_use:
                summary_parts.append(f"Alcohol use: {self.lifestyle.alcohol_use}.")
        
        return " ".join(summary_parts)


# Example usage and parsing helper
class FHIRToPatientParser:
    """Helper class to parse FHIR resources into Patient model"""
    
    @staticmethod
    def parse_fhir_bundle(fhir_bundle: dict) -> Patient:
        """
        Parse a FHIR Bundle into a Patient object.
        This is a skeleton - you'll need to implement the actual parsing logic.
        """
        # This is where you'd implement the actual FHIR parsing
        # Example structure:
        patient_resource = None
        conditions = []
        medications = []
        observations = []
        procedures = []
        allergies = []
        
        for entry in fhir_bundle.get("entry", []):
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")
            
            if resource_type == "Patient":
                patient_resource = resource
            elif resource_type == "Condition":
                conditions.append(resource)
            elif resource_type == "MedicationRequest":
                medications.append(resource)
            elif resource_type == "Observation":
                observations.append(resource)
            elif resource_type == "Procedure":
                procedures.append(resource)
            elif resource_type == "AllergyIntolerance":
                allergies.append(resource)
        
        # Now you'd parse each resource into the appropriate Pydantic model
        # This is pseudocode - actual implementation depends on FHIR structure
        
        return Patient(
            demographics=Demographics(
                patient_id="placeholder",
                date_of_birth=date_type.today(),
                age=0,
                gender=Gender.UNKNOWN
            )
        )