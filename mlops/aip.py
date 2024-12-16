from  google.cloud import aiplatform

class Aiplatforminit:
    def __init__(self, project_id: str, location: str, staging_bucket: str):
        self.project_id = project_id
        self.location = location
        self.staging_bucket = staging_bucket

        # Initialize Vertex AI
        aiplatform.init(project=self.project_id, location=self.location)