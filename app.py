import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from PyPDF2 import PdfReader

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="AI Resume & ATS Analyzer", layout="wide")

# ------------------- SIDEBAR -------------------
page = st.sidebar.selectbox("Choose Tool", ["AI Resume Generator", "ATS Analyzer"])
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])

# ------------------- THEME HANDLER -------------------
def set_theme(theme):
    if theme == "Dark":
        st.markdown(
            """
            <style>
            body { background-color: #121212; color: #ffffff; }
            .stApp { background-color: #121212; }
            .stTextInput > div > div > input, 
            .stTextArea > div > div > textarea {
                background-color: #1e1e1e;
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            body { background-color: #ffffff; color: #000000; }
            .stApp { background-color: #ffffff; }
            .stTextInput > div > div > input, 
            .stTextArea > div > div > textarea {
                background-color: #f9f9f9;
                color: black;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

set_theme(theme)

# ------------------- AI RESUME GENERATOR -------------------
if page == "AI Resume Generator":
    @st.cache_resource
    def load_model():
        model_name = "google/flan-t5-base"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
        return tokenizer, model

    tokenizer, model = load_model()

    def generate_text(prompt, max_length=512):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def create_resume_html(name, email, phone, role, skills, experience, goals, education):
        resume_html = f"""
        <div style="
            background-color: #f9f9f9;
            border: 2px solid #4CAF50;
            border-radius: 12px;
            padding: 20px;
            font-family: Arial, sans-serif;
            color: #333;
            line-height: 1.6;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <h3>Here is the cleaned-up resume:</h3>
            <p><strong>{name}</strong></p>
            <p><strong>Email:</strong> <a href="mailto:{email}">{email}</a></p>
            <p><strong>Phone:</strong> {phone}</p>
            <p><strong>Objective:</strong> {goals}</p>
            <p><strong>Technical Skills:</strong> {skills}</p>
            <p><strong>Education:</strong> {education}</p>
            <p><strong>Projects & Experience:</strong> {experience}</p>
        </div>
        """
        return resume_html

    def create_cover_letter_html(name, email, phone, role, skills, experience, goals, education):
        cover_letter_text = f"""
        Dear Hiring Manager,<br><br>
        I am writing to express my interest in the position of {role}. As a graduate with a strong foundation in {skills}, I have developed the ability to solve problems creatively and deliver efficient solutions. My background in {education} has provided me with both technical expertise and the determination to adapt to new challenges.<br><br>
        I have successfully worked on projects that strengthened my knowledge of software development, coding, and debugging. My passion for technology drives me to keep learning, improving, and staying updated with modern industry practices. I believe my enthusiasm and dedication make me a strong fit for this role.<br><br>
        Thank you for taking the time to review my application. I look forward to the opportunity to bring my skills and passion to your team and discuss how I can contribute to your organization’s success.<br><br>
        Sincerely,<br>
        {name}
        """
        cover_html = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; color: #333; border: 2px solid #ccc; border-radius: 10px; line-height: 1.6;">
            <h2 style="color:#2c3e50;">Cover Letter</h2>
            <p>{cover_letter_text}</p>
        </div>
        """
        return cover_html

    st.title("📄 AI Resume & Cover Letter Generator")
    with st.form("resume_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone Number")
        role = st.text_input("Job Role (e.g., Software Engineer)")
        skills = st.text_area("Skills (comma-separated)")
        experience = st.text_area("Work Experience/Projects (e.g., 2 years in backend development...)")
        goals = st.text_area("Career Objective / Goal")
        education = st.text_area("Education")
        submitted = st.form_submit_button("Generate Resume & Cover Letter")

    if submitted:
        with st.spinner("Generating your professional resume and cover letter..."):
            resume_html = create_resume_html(name, email, phone, role, skills, experience, goals, education)
            cover_html = create_cover_letter_html(name, email, phone, role, skills, experience, goals, education)

        st.subheader("🧾 **Professional Resume**")
        st.components.v1.html(resume_html, height=400, scrolling=True)
        st.subheader("✉️ **Professional Cover Letter**")
        st.components.v1.html(cover_html, height=500, scrolling=True)
        combined = f"--- RESUME ---\n\n{resume_html}\n\n--- COVER LETTER ---\n\n{cover_html}"
        st.download_button("⬇️ Download as HTML", combined, file_name="AI_Resume_Cover_Letter.html")

# ------------------- ATS ANALYZER -------------------
elif page == "ATS Analyzer":
    st.title("ATS Resume Analyzer")
    job_skills_input = st.text_area("Enter required skills (comma separated):", "Python, SQL, Machine Learning")
    job_skills = [skill.strip() for skill in job_skills_input.split(",")]

    uploaded_files = st.file_uploader(
        "Upload resumes (.txt, .pdf, .html files)",
        type=["txt", "pdf", "html"],
        accept_multiple_files=True
    )

    def extract_text_from_file(uploaded_file):
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        elif uploaded_file.type == "text/html":
            return uploaded_file.read().decode("utf-8")
        else:
            return uploaded_file.read().decode("utf-8")

    def calculate_match(resume_text, skills):
        score = sum(1 for skill in skills if skill.lower() in resume_text.lower())
        return (score / len(skills)) * 100

    if st.button("Analyze Resumes"):
        if uploaded_files:
            results = []
            for uploaded_file in uploaded_files:
                content = extract_text_from_file(uploaded_file)
                score = calculate_match(content, job_skills)
                results.append((uploaded_file.name, score))
            results.sort(key=lambda x: x[1], reverse=True)
            st.subheader("Results:")
            for resume, score in results:
                st.write(f"**{resume}**: {score:.2f}% match")
        else:
            st.warning("Please upload at least one resume file.")
