"""
In Infra coding, its better to have a set of reated resources to be defined within the same stack.
In this code, we habve all the patient related infra setup under the same stack.
"""
import pulumi
import pulumi_gcp as gcp
from pulumi import Config, export
from typing import Optional

class PatientStack:
    def __init__(self,name:str,project_id:str,region:str="us-central1",opts:Optional[pulumi.ResourceOptions]=None):
        self.name = name 
        self.project_id = project_id 
        self.region = region 
        self.opts = opts or pulumi.ResourceOptions()

        self.firestore_db = self._create_firestore()
        self._export_outputs()
        
    def _create_firestore(self) -> gcp.firestore.Database:
        return gcp.firestore.Database(
            f"{self.name}-patient-database",
            project = self.project_id,
            name=f"patient-db-{self.name}",
            location_id = self.region,
            type = "FIRESTORE_NATIVE",
            concurrency_mode = "OPTIMISTIC",
            opts = self.opts
        )
    
    def _export_outputs(self):
        pulumi.export(f"{self.name}_firestore_db",self.firestore_db.name)


