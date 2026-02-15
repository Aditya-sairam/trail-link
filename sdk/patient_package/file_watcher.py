import time
import json
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from data_parser import parse_fhir_file
from typing import Dict
import shutil
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FHIRFileHandler(FileSystemEventHandler):
    """
    Handles new FHIR JSON files automatically.
    Watches the incoming directory and processes files as they arrive.
    """
    
    def __init__(self, patients_db: Dict):
        self.patients_db = patients_db
        
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        
        # Define folder paths
        self.incoming_dir = script_dir / "data" / "incoming"
        self.processed_dir = script_dir / "data" / "processed"
        self.failed_dir = script_dir / "data" / "failed"
        
        # Create directories if they don't exist
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
    
    def on_created(self, event):
        """Called when a new file is created."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only process JSON files
        if file_path.suffix.lower() != '.json':
            logger.info(f"Ignoring non-JSON file: {file_path.name}")
            return
        
        # Ignore .gitkeep
        if file_path.name == '.gitkeep':
            return
        
        logger.info(f"🔔 New file detected: {file_path.name}")
        
        # Wait for file to be fully written
        time.sleep(1)
        
        # Process the file
        self.process_fhir_file(file_path)
    
    def process_fhir_file(self, file_path: Path):
        """Parse FHIR file and add patient to database."""
        try:
            logger.info(f"📄 Processing file: {file_path.name}")
            
            # Parse the FHIR file
            patient = parse_fhir_file(str(file_path))
            
            # Add to database
            patient_id = patient.demographics.patient_id
            self.patients_db[patient_id] = patient
            
            logger.info(f"✅ Successfully parsed patient: {patient_id}")
            logger.info(f"📊 Total patients in database: {len(self.patients_db)}")
            
            # Move to processed directory
            destination = self.processed_dir / file_path.name
            shutil.move(str(file_path), str(destination))
            logger.info(f"📁 Moved file to: {destination}")
            
            # Save parsed data
            parsed_file = self.processed_dir / f"parsed_{file_path.stem}.json"
            with open(parsed_file, 'w') as f:
                json.dump(patient.model_dump(), f, indent=2, default=str)
            logger.info(f"💾 Saved parsed data to: {parsed_file}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {str(e)}")
            
            # Move to failed directory
            destination = self.failed_dir / file_path.name
            try:
                shutil.move(str(file_path), str(destination))
                logger.info(f"📁 Moved failed file to: {destination}")
            except Exception as move_error:
                logger.error(f"Could not move failed file: {move_error}")


def start_file_watcher(patients_db: Dict):
    """Start watching the incoming directory for new files."""
    script_dir = Path(__file__).parent
    incoming_dir = script_dir / "data" / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    
    event_handler = FHIRFileHandler(patients_db)
    observer = Observer()
    observer.schedule(event_handler, str(incoming_dir), recursive=False)
    observer.start()
    
    logger.info(f"👀 File watcher started!")
    logger.info(f"📂 Monitoring directory: {incoming_dir.absolute()}")
    logger.info(f"💡 Drop FHIR JSON files into this directory to auto-process them")
    
    return observer
