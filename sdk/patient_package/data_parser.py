import json
from datetime import datetime,timedelta, date 
from typing import Dict, Any, List, Optional 
from pathlib import Path
from data_models import Allergy, Condition, ConditionStatus, Demographics,Gender, LifestyleFactors, Medication, Observation, Patient, Procedure


class FHIRParser:
    def __init__(self,json_file_path:str): 
        self.json_file_path = json_file_path
        self.fhir_bundle = None 
        self.resources = {} 

    def load_json(self) -> Dict[str, Any]: 
        with open(self.json_file_path,'r') as f:
            self.fhir_bundle = json.load(f)
        return self.fhir_bundle 
    
    def organize_resources(self) -> Dict[str,Any]:
        if not self.fhir_bundle:
            raise ValueError("FHIR bundle not loaded. load json failed for some reason.")
        
        entries =self.fhir_bundle.get("entry",[])

        for entry in entries:
            resource = entry.get("resource",{})
            resource_type = resource.get("resourceType","Unknown")
            if resource_type not in self.resources:
                self.resources[resource_type] = []
            self.resources[resource_type].append(resource)
    
    def parse_date(self,date_string:Optional[str]) -> Optional[date]: 
        if not date_string:
            return None 
        try: 
            if len(date_string) == 4: 
                return datetime.strptime(date_string,"%Y").date() 
            elif len(date_string) == 7:
                return datetime.strptime(date_string,"%Y-%m").date()
            elif len(date_string) == 10:
                return datetime.strptime(date_string,"%Y-%m-%d").date()
            else:
                return datetime.strptime(date_string[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
        
    def calc_age(self,birth_date:date) -> int:
        today = date.today()
        age = today.year - birth_date.year 
        if (today.month,today.day) < (birth_date.month,birth_date.day):
            age -= 1
        return age 
    
    def parse_demographics(self) -> Demographics:
        patient_resources = self.resources.get("Patient",[])
        if not patient_resources:
            raise ValueError("No Patient resource found in FHIR data.")
        
        patient = patient_resources[0]

        patient_id = patient.get("id","Unknown")
        birth_date_str = patient.get("birthDate")
        birth_date = self.parse_date(birth_date_str)
        age = self.calc_age(birth_date) if birth_date else -1

        gender_str = patient.get("gender","unknown").lower() 
        gender = Gender(gender_str)if gender_str in ["male", "female", "other"] else Gender.UNKNOWN
    
        race = None 
        extensions = patient.get("extension",[])

        for ext in extensions:
            if "us-core-race" in ext.get("url",""):
                for sub_ext in ext.get("extension",[]):
                    if sub_ext.get("url") == "text":
                        race = sub_ext.get("valueString")
                        break 
        marital_status = None
        marital_status_obj = patient.get("maritalStatus", {})
        if marital_status_obj:
            coding = marital_status_obj.get("coding", [{}])[0]
            marital_status = coding.get("display")
        
        # Extract address
        addresses = patient.get("address", [])
        city, state, country = None, None, None
        if addresses:
            addr = addresses[0]
            city = addr.get("city")
            state = addr.get("state")
            country = addr.get("country")
        
        return Demographics(
            patient_id=patient_id,
            date_of_birth=birth_date,
            age=age,
            gender=gender,
            race=race,
            marital_status=marital_status,
            city=city,
            state=state,
            country=country
        )
    
    def parse_conditions(self) -> List[Condition]:
        """Parse Condition resources into Condition models"""
        condition_resources = self.resources.get("Condition", [])
        conditions = []
        
        for cond in condition_resources:
            # Get code and display name
            code_obj = cond.get("code", {})
            coding = code_obj.get("coding", [{}])[0]
            code = coding.get("code", "unknown")
            display_name = coding.get("display", code_obj.get("text", "Unknown condition"))
            
            # Parse dates
            onset_date = self.parse_date(cond.get("onsetDateTime"))
            abatement_date = self.parse_date(cond.get("abatementDateTime"))
            
            # Parse status
            clinical_status = cond.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "active")
            status = ConditionStatus.ACTIVE
            if clinical_status == "resolved":
                status = ConditionStatus.RESOLVED
            elif clinical_status == "remission":
                status = ConditionStatus.REMISSION
            elif clinical_status == "recurrence":
                status = ConditionStatus.RECURRENCE
            
            conditions.append(Condition(
                code=code,
                display_name=display_name,
                onset_date=onset_date,
                abatement_date=abatement_date,
                status=status
            ))
        
        return conditions
    
    def parse_medications(self) -> List[Medication]:
        """Parse MedicationRequest resources into Medication models"""
        med_resources = self.resources.get("MedicationRequest", [])
        medications = []
        
        for med in med_resources:
            # Get medication code and name
            med_code_obj = med.get("medicationCodeableConcept", {})
            coding = med_code_obj.get("coding", [{}])[0]
            code = coding.get("code")
            display_name = coding.get("display", med_code_obj.get("text", "Unknown medication"))
            
            # Get dosage
            dosage_instructions = med.get("dosageInstruction", [])
            dosage = None
            route = None
            if dosage_instructions:
                dosage_info = dosage_instructions[0]
                dosage_text = dosage_info.get("text")
                route = dosage_info.get("route", {}).get("text")
                dosage = dosage_text
            
            # Get dates
            start_date = self.parse_date(med.get("authoredOn"))
            
            # Get status
            status = med.get("status", "active")
            
            # Get reason
            reason_code = med.get("reasonCode", [])
            reason = reason_code[0].get("text") if reason_code else None
            
            medications.append(Medication(
                code=code,
                display_name=display_name,
                dosage=dosage,
                route=route,
                start_date=start_date,
                status=status,
                reason=reason
            ))
        
        return medications
    
    def parse_observations(self) -> List[Observation]:
        """Parse Observation resources into Observation models"""
        obs_resources = self.resources.get("Observation", [])
        observations = []
        
        for obs in obs_resources:
            # Get code
            code_obj = obs.get("code", {})
            coding = code_obj.get("coding", [{}])[0]
            code = coding.get("code", "unknown")
            display_name = coding.get("display", code_obj.get("text", "Unknown observation"))
            
            # Get value
            value = None
            value_string = None
            unit = None
            
            if "valueQuantity" in obs:
                value_qty = obs["valueQuantity"]
                value = value_qty.get("value")
                unit = value_qty.get("unit")
            elif "valueString" in obs:
                value_string = obs["valueString"]
            elif "valueCodeableConcept" in obs:
                value_string = obs["valueCodeableConcept"].get("text")
            
            # Get reference range
            ref_range_low = None
            ref_range_high = None
            ref_ranges = obs.get("referenceRange", [])
            if ref_ranges:
                ref_range = ref_ranges[0]
                if "low" in ref_range:
                    ref_range_low = ref_range["low"].get("value")
                if "high" in ref_range:
                    ref_range_high = ref_range["high"].get("value")
            
            # Get date
            obs_date = self.parse_date(obs.get("effectiveDateTime"))
            if not obs_date:
                obs_date = date.today()
            
            # Get category
            categories = obs.get("category", [])
            category = None
            if categories:
                category = categories[0].get("coding", [{}])[0].get("code")
            
            observations.append(Observation(
                code=code,
                display_name=display_name,
                value=value,
                value_string=value_string,
                unit=unit,
                reference_range_low=ref_range_low,
                reference_range_high=ref_range_high,
                date=obs_date,
                category=category
            ))
        
        return observations
    
    def parse_procedures(self) -> List[Procedure]:
        """Parse Procedure resources into Procedure models"""
        proc_resources = self.resources.get("Procedure", [])
        procedures = []
        
        for proc in proc_resources:
            # Get code
            code_obj = proc.get("code", {})
            coding = code_obj.get("coding", [{}])[0]
            code = coding.get("code", "unknown")
            display_name = coding.get("display", code_obj.get("text", "Unknown procedure"))
            
            # Get performed date
            performed_date = self.parse_date(proc.get("performedDateTime"))
            if not performed_date:
                performed_period = proc.get("performedPeriod", {})
                performed_date = self.parse_date(performed_period.get("start"))
            
            if not performed_date:
                performed_date = date.today()
            
            # Get body site
            body_site = None
            body_site_obj = proc.get("bodySite", [])
            if body_site_obj:
                body_site = body_site_obj[0].get("text")
            
            # Get reason
            reason_code = proc.get("reasonCode", [])
            reason = reason_code[0].get("text") if reason_code else None
            
            procedures.append(Procedure(
                code=code,
                display_name=display_name,
                performed_date=performed_date,
                body_site=body_site,
                reason=reason
            ))
        
        return procedures
    
    def parse_allergies(self) -> List[Allergy]:
        """Parse AllergyIntolerance resources into Allergy models"""
        allergy_resources = self.resources.get("AllergyIntolerance", [])
        allergies = []
        
        for allergy in allergy_resources:
            # Get substance
            code_obj = allergy.get("code", {})
            substance = code_obj.get("text", "Unknown substance")
            
            # Get criticality
            criticality = allergy.get("criticality")
            
            # Get reactions
            reactions = []
            reaction_list = allergy.get("reaction", [])
            for reaction in reaction_list:
                manifestations = reaction.get("manifestation", [])
                for manifestation in manifestations:
                    reactions.append(manifestation.get("text", ""))
            
            # Get onset date
            onset_date = self.parse_date(allergy.get("onsetDateTime"))
            
            # Get type
            allergy_type = allergy.get("type", "allergy")
            
            allergies.append(Allergy(
                substance=substance,
                criticality=criticality,
                reaction=reactions if reactions else None,
                onset_date=onset_date,
                type=allergy_type
            ))
        
        return allergies

    def parse_lifestyle(self) -> Optional[LifestyleFactors]:
        """Parse lifestyle factors from Observation resources"""
        obs_resources = self.resources.get("Observation", [])
        
        smoking_status = None
        alcohol_use = None
        
        for obs in obs_resources:
            code_obj = obs.get("code", {})
            coding = code_obj.get("coding", [{}])[0]
            code = coding.get("code", "")
            
            # Smoking status (LOINC code 72166-2)
            if code == "72166-2":
                value_obj = obs.get("valueCodeableConcept", {})
                smoking_status = value_obj.get("text")
            
            # Could add alcohol use parsing here if available
        
        if smoking_status or alcohol_use:
            return LifestyleFactors(
                smoking_status=smoking_status,
                alcohol_use=alcohol_use
            )
        
        return None
    
    def parse_to_patient(self) -> Patient:
        """
        Main method: Parse entire FHIR bundle into Patient model
        
        Returns:
            Patient: Complete patient object with all parsed data
        """
        # Load JSON if not already loaded
        if not self.fhir_bundle:
            self.load_json()
        
        # Organize resources
        self.organize_resources()
        
        # Parse each component
        demographics = self.parse_demographics()
        conditions = self.parse_conditions()
        medications = self.parse_medications()
        observations = self.parse_observations()
        procedures = self.parse_procedures()
        allergies = self.parse_allergies()
        lifestyle = self.parse_lifestyle()
        
        # Create and return Patient object
        return Patient(
            demographics=demographics,
            conditions=conditions,
            medications=medications,
            observations=observations,
            procedures=procedures,
            allergies=allergies,
            lifestyle=lifestyle
        )

def parse_fhir_file(json_file_path: str) -> Patient:
    """
    Convenience function to parse a FHIR JSON file into a Patient object
    
    Args:
        json_file_path: Path to the Synthea FHIR JSON file
        
    Returns:
        Patient: Parsed patient data
    """
    parser = FHIRParser(json_file_path)
    return parser.parse_to_patient()


# Main function, just to test the parser locally.
if __name__ == "__main__":
    # Example usage
    file_path = "/home/aditya/MlOps/trial-link/data/output/fhir/Lan153_Junko239_White193_11b21749-2d42-a3ae-5720-60ebcbbdb194.json"
    
    try:
        patient = parse_fhir_file(file_path)
        
        # Print summary
        print("Successfully parsed patient data!")
        print(f"\nPatient ID: {patient.demographics.patient_id}")
        print(f"Patient location: {patient.demographics.city}, {patient.demographics.state}, {patient.demographics.country}")
        print(f"Patient marital status: {patient.demographics.marital_status}")
        print(f"Age: {patient.demographics.age}")
        print(f"Gender: {patient.demographics.gender.value}")
        print(f"\nActive Conditions: {len(patient.get_active_conditions())}")
        print(f"Current Medications: {len(patient.get_current_medications())}")
        print(f"Total Observations: {len(patient.observations)}")
        print(f"Procedures: {len(patient.procedures)}")
        print(f"Allergies: {len(patient.allergies)}")
        
        # Print text summary for RAG
        print("\n" + "="*50)
        print("TEXT SUMMARY FOR RAG SYSTEM:")
        print("="*50)
        print(patient.to_text_summary())
        
        # Optional: Save as JSON
        with open("parsed_patient.json", "w") as f:
            json.dump(patient.model_dump(), f, indent=2, default=str)
        print("\nParsed data saved to parsed_patient.json")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error parsing file: {e}")
        import traceback
        traceback.print_exc()
   



