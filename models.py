"""Database tables (SQLAlchemy). SQLite is used by default."""
import re
from datetime import datetime

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    full_name = db.Column(db.String(100), nullable=False, default="")
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(10), nullable=False, default="teacher")  # admin | teacher
    active_flag = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def is_active(self):  # used by Flask-Login
        return self.active_flag

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Exam(db.Model):
    __tablename__ = "exams"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150), nullable=False)
    subject = db.Column(db.String(100), nullable=False, default="")
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    questions = db.relationship(
        "Question", backref="exam", cascade="all, delete-orphan", order_by="Question.number"
    )
    sheets = db.relationship(
        "AnswerSheet", backref="exam", cascade="all, delete-orphan", order_by="AnswerSheet.roll_no"
    )

    @property
    def total_marks(self):
        return sum(q.max_marks for q in self.questions)


class Question(db.Model):
    __tablename__ = "questions"
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey("exams.id"), nullable=False)
    number = db.Column(db.Integer, nullable=False)
    max_marks = db.Column(db.Float, nullable=False)
    model_answer = db.Column(db.Text, nullable=False)
    keywords = db.Column(db.Text, nullable=False, default="")  # separated by ; or |

    @property
    def keyword_list(self):
        return [k.strip() for k in re.split(r"[;|]", self.keywords or "") if k.strip()]


class AnswerSheet(db.Model):
    __tablename__ = "answer_sheets"
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey("exams.id"), nullable=False)
    student_name = db.Column(db.String(100), nullable=False)
    roll_no = db.Column(db.String(30), nullable=False)
    filename = db.Column(db.String(255), nullable=False)      # name the teacher uploaded
    stored_name = db.Column(db.String(255), nullable=False)   # name on disk
    status = db.Column(db.String(20), nullable=False, default="uploaded")
    # uploaded | processing | evaluated | reviewed | finalized | failed
    ocr_engine = db.Column(db.String(20), default="")
    raw_text = db.Column(db.Text, default="")
    error_message = db.Column(db.String(500), default="")
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    finalized_at = db.Column(db.DateTime)

    results = db.relationship(
        "AnswerResult", backref="sheet", cascade="all, delete-orphan", order_by="AnswerResult.question_no"
    )

    @property
    def total_final(self):
        return sum((r.final_marks or 0) for r in self.results)

    @property
    def total_ai(self):
        return sum((r.ai_marks or 0) for r in self.results)


class AnswerResult(db.Model):
    __tablename__ = "answer_results"
    id = db.Column(db.Integer, primary_key=True)
    sheet_id = db.Column(db.Integer, db.ForeignKey("answer_sheets.id"), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey("questions.id"), nullable=False)
    question_no = db.Column(db.Integer, nullable=False)
    extracted_text = db.Column(db.Text, default="")
    similarity = db.Column(db.Float, default=0.0)        # 0 to 1
    keyword_score = db.Column(db.Float, default=0.0)     # 0 to 1
    ai_marks = db.Column(db.Float, default=0.0)
    final_marks = db.Column(db.Float, default=0.0)       # teacher-approved marks
    feedback = db.Column(db.Text, default="")            # automatic explanation
    remarks = db.Column(db.Text, default="")             # teacher remark
    reviewed = db.Column(db.Boolean, default=False)

    question = db.relationship("Question")
