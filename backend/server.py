from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime
import asyncio

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class TherapyAreaRequest(BaseModel):
    therapy_area: str
    product_name: Optional[str] = None
    api_key: str

class TherapyAreaAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    therapy_area: str
    product_name: Optional[str] = None
    disease_summary: str
    staging: str
    biomarkers: str
    treatment_algorithm: str
    patient_journey: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PatientFlowFunnelRequest(BaseModel):
    therapy_area: str
    analysis_id: str
    api_key: str

class PatientFlowFunnel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    therapy_area: str
    analysis_id: str
    funnel_stages: List[dict]
    total_addressable_population: str
    forecasting_notes: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Pharma Forecasting Consultant API"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

@api_router.post("/analyze-therapy", response_model=TherapyAreaAnalysis)
async def analyze_therapy_area(request: TherapyAreaRequest):
    try:
        # Import Claude integration
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        
        # Create Claude chat instance
        chat = LlmChat(
            api_key=request.api_key,
            session_id=f"therapy_analysis_{uuid.uuid4()}",
            system_message="""You are a world-class pharmaceutical consultant specializing in therapy area analysis and forecasting. 
            You have deep expertise in disease pathology, treatment algorithms, biomarkers, and patient journey mapping.
            Provide comprehensive, accurate, and structured analysis suitable for pharmaceutical forecasting models."""
        ).with_model("anthropic", "claude-sonnet-4-20250514").with_max_tokens(4096)
        
        # Construct analysis prompt
        product_info = f" for the product '{request.product_name}'" if request.product_name else ""
        prompt = f"""
        Please provide a comprehensive analysis of the {request.therapy_area} therapy area{product_info}. 
        
        Structure your response in exactly 5 sections with clear headers:
        
        ## DISEASE SUMMARY
        [Provide overview of the disease/condition, epidemiology, prevalence, and key clinical characteristics]
        
        ## STAGING
        [Detail the disease staging system, progression stages, and clinical classifications used]
        
        ## BIOMARKERS
        [List key biomarkers, diagnostic markers, prognostic indicators, and companion diagnostics]
        
        ## TREATMENT ALGORITHM
        [Describe current treatment pathways, standard of care, decision points, and treatment sequencing]
        
        ## PATIENT JOURNEY
        [Map the complete patient journey from symptoms to diagnosis to treatment and follow-up care]
        
        Focus on current medical standards and include relevant clinical data where appropriate.
        """
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        # Parse the response into structured sections
        sections = response.split("## ")
        disease_summary = ""
        staging = ""
        biomarkers = ""
        treatment_algorithm = ""
        patient_journey = ""
        
        for section in sections[1:]:  # Skip first empty element
            if section.startswith("DISEASE SUMMARY"):
                disease_summary = section.replace("DISEASE SUMMARY\n", "").strip()
            elif section.startswith("STAGING"):
                staging = section.replace("STAGING\n", "").strip()
            elif section.startswith("BIOMARKERS"):
                biomarkers = section.replace("BIOMARKERS\n", "").strip()
            elif section.startswith("TREATMENT ALGORITHM"):
                treatment_algorithm = section.replace("TREATMENT ALGORITHM\n", "").strip()
            elif section.startswith("PATIENT JOURNEY"):
                patient_journey = section.replace("PATIENT JOURNEY\n", "").strip()
        
        # Create analysis object
        analysis = TherapyAreaAnalysis(
            therapy_area=request.therapy_area,
            product_name=request.product_name,
            disease_summary=disease_summary,
            staging=staging,
            biomarkers=biomarkers,
            treatment_algorithm=treatment_algorithm,
            patient_journey=patient_journey
        )
        
        # Save to database
        await db.therapy_analyses.insert_one(analysis.dict())
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in therapy analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_router.post("/generate-funnel", response_model=PatientFlowFunnel)
async def generate_patient_flow_funnel(request: PatientFlowFunnelRequest):
    try:
        # Get the original analysis
        analysis = await db.therapy_analyses.find_one({"id": request.analysis_id})
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Import Claude integration
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        
        # Create Claude chat instance
        chat = LlmChat(
            api_key=request.api_key,
            session_id=f"funnel_generation_{uuid.uuid4()}",
            system_message="""You are a pharmaceutical forecasting expert specializing in patient flow modeling and market analysis.
            Create detailed patient flow funnels suitable for pharmaceutical forecasting models based on therapy area analysis."""
        ).with_model("anthropic", "claude-sonnet-4-20250514").with_max_tokens(4096)
        
        # Construct funnel generation prompt
        prompt = f"""
        Based on the following therapy area analysis for {request.therapy_area}, create a comprehensive patient flow funnel suitable for pharmaceutical forecasting:
        
        THERAPY AREA: {request.therapy_area}
        DISEASE SUMMARY: {analysis['disease_summary'][:500]}...
        TREATMENT ALGORITHM: {analysis['treatment_algorithm'][:500]}...
        PATIENT JOURNEY: {analysis['patient_journey'][:500]}...
        
        Please provide your response in exactly this JSON structure:
        
        {{
            "funnel_stages": [
                {{
                    "stage": "Total Population at Risk",
                    "description": "Overall population that could develop this condition",
                    "percentage": "100%",
                    "notes": "Base population estimates"
                }},
                {{
                    "stage": "Disease Incidence/Prevalence",
                    "description": "Population that develops or has the condition",
                    "percentage": "X%",
                    "notes": "Epidemiological data"
                }},
                {{
                    "stage": "Diagnosis Rate",
                    "description": "Patients who get properly diagnosed",
                    "percentage": "X%",
                    "notes": "Diagnosis challenges and rates"
                }},
                {{
                    "stage": "Treatment Eligible",
                    "description": "Diagnosed patients eligible for treatment",
                    "percentage": "X%",
                    "notes": "Contraindications and eligibility criteria"
                }},
                {{
                    "stage": "Treated Patients",
                    "description": "Patients actually receiving treatment",
                    "percentage": "X%",
                    "notes": "Treatment uptake and access"
                }},
                {{
                    "stage": "Target Patient Population",
                    "description": "Specific target for your therapy/product",
                    "percentage": "X%",
                    "notes": "Specific targeting criteria"
                }}
            ],
            "total_addressable_population": "Detailed TAM analysis with numbers and rationale",
            "forecasting_notes": "Key assumptions, market dynamics, competitive landscape considerations, and forecasting methodology recommendations"
        }}
        
        Fill in realistic percentages and detailed descriptions based on current medical literature and market data for {request.therapy_area}.
        """
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        # Parse JSON response
        import json
        try:
            # Clean the response to extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            parsed_response = json.loads(json_str)
        except:
            # Fallback parsing if JSON extraction fails
            parsed_response = {
                "funnel_stages": [
                    {"stage": "Total Population", "description": "Analysis generated", "percentage": "100%", "notes": "See full response"},
                    {"stage": "Target Population", "description": "Detailed analysis provided", "percentage": "Variable", "notes": response[:200] + "..."}
                ],
                "total_addressable_population": "See full analysis response",
                "forecasting_notes": response
            }
        
        # Create funnel object
        funnel = PatientFlowFunnel(
            therapy_area=request.therapy_area,
            analysis_id=request.analysis_id,
            funnel_stages=parsed_response.get("funnel_stages", []),
            total_addressable_population=parsed_response.get("total_addressable_population", ""),
            forecasting_notes=parsed_response.get("forecasting_notes", "")
        )
        
        # Save to database
        await db.patient_flow_funnels.insert_one(funnel.dict())
        
        return funnel
        
    except Exception as e:
        logger.error(f"Error in funnel generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Funnel generation failed: {str(e)}")

@api_router.get("/analyses", response_model=List[TherapyAreaAnalysis])
async def get_therapy_analyses():
    analyses = await db.therapy_analyses.find().sort("created_at", -1).to_list(50)
    return [TherapyAreaAnalysis(**analysis) for analysis in analyses]

@api_router.get("/funnels/{analysis_id}")
async def get_funnel_by_analysis(analysis_id: str):
    funnel = await db.patient_flow_funnels.find_one({"analysis_id": analysis_id})
    if not funnel:
        return None
    return PatientFlowFunnel(**funnel)

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()