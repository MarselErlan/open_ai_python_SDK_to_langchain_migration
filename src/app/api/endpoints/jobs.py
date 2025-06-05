from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import json
from src.app.db.session import get_db
from src.app.models.models import JobPosting as JobPostingModel, Company
from src.app.schemas.schemas import JobPosting as JobPostingSchema, JobPostingCreate, JobPostingUpdate, JobDescriptionRequest, JobDescriptionResponse, JobDescriptionComponent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from src.app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize LangChain chat model
chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=500,
    model_kwargs={"top_p": 0.9}
)

def get_db_session():
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()

# Job CRUD endpoints (unchanged)
@router.post("/", response_model=JobPostingSchema)
def create_job_posting(job: JobPostingCreate, db: Session = Depends(get_db_session)):
    company = db.query(Company).filter(Company.id == job.company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    db_job = JobPostingModel(**job.model_dump())
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job

@router.get("/", response_model=List[JobPostingSchema])
def read_job_postings(skip: int = 0, limit: int = 100, db: Session = Depends(get_db_session)):
    jobs = db.query(JobPostingModel).offset(skip).limit(limit).all()
    return jobs

@router.get("/{job_id}", response_model=JobPostingSchema)
def read_job_posting(job_id: int, db: Session = Depends(get_db_session)):
    db_job = db.query(JobPostingModel).filter(JobPostingModel.id == job_id).first()
    if db_job is None:
        raise HTTPException(status_code=404, detail="Job posting not found")
    return db_job

@router.put("/{job_id}", response_model=JobPostingSchema)
def update_job_posting(job_id: int, job: JobPostingUpdate, db: Session = Depends(get_db_session)):
    db_job = db.query(JobPostingModel).filter(JobPostingModel.id == job_id).first()
    if db_job is None:
        raise HTTPException(status_code=404, detail="Job posting not found")
    
    if job.company_id is not None:
        company = db.query(Company).filter(Company.id == job.company_id).first()
        if not company:
            raise HTTPException(status_code=404, detail="Company not found")
    
    update_data = job.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_job, field, value)
    
    db.commit()
    db.refresh(db_job)
    return db_job

@router.delete("/{job_id}")
def delete_job_posting(job_id: int, db: Session = Depends(get_db_session)):
    db_job = db.query(JobPostingModel).filter(JobPostingModel.id == job_id).first()
    if db_job is None:
        raise HTTPException(status_code=404, detail="Job posting not found")
    
    db.delete(db_job)
    db.commit()
    return {"message": "Job posting deleted successfully"}

# Job Description Generation with LangChain
@router.post("/{id}/description", response_model=JobDescriptionResponse)
async def generate_job_description(id: int, request: JobDescriptionRequest, db: Session = Depends(get_db_session)):
    # Retrieve the job posting and associated company
    job = db.query(JobPostingModel).filter(JobPostingModel.id == id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job posting not found")

    if not job.company:
        raise HTTPException(status_code=404, detail="Associated company not found")

    # Set up Pydantic output parser
    parser = PydanticOutputParser(pydantic_object=JobDescriptionComponent)

    # Define ChatPromptTemplate
    system_template = """
    You are a professional HR assistant specializing in writing job descriptions. 
    Generate a structured job description based on the provided information. 
    Follow the format specified by the output parser.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", (
            "Job Title: {title}\n"
            "Company: {company_name}, {industry}\n"
            "Required Tools: {tools}\n"
            "Company Culture: {culture}\n"
            "Provide a detailed job description with title, overview, responsibilities, requirements, qualifications, and benefits."
        ))
    ])

    # Prepare prompt variables
    tools = ", ".join(request.required_tools)
    culture = request.company_culture or "professional and collaborative environment"
    prompt = prompt_template.format_messages(
        title=job.title,
        company_name=job.company.name,
        industry=job.company.industry,
        tools=tools,
        culture=culture
    )

    try:
        # Generate structured output with LangChain
        response = chat_model.with_structured_output(JobDescriptionComponent).invoke(prompt)
        
        # Save the description to the database
        job.description = json.dumps(response.model_dump())
        db.commit()
        db.refresh(job)

        # Prepare the response
        generated_at = datetime.utcnow()
        return JobDescriptionResponse(
            job_id=job.id,
            description=response,
            generated_at=generated_at
        )

    except Exception as e:
        logger.error(f"Error generating job description: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate job description: {str(e)}")