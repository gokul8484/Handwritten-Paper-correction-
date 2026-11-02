"""Module 1 and 7: Flask web application (login, upload, review, reports)."""
import os
import uuid
from datetime import datetime
from functools import wraps

import click
from flask import (Flask, abort, flash, redirect, render_template, request,
                   send_file, url_for)
from flask_login import (LoginManager, current_user, login_required, login_user,
                         logout_user)
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename

import pipeline
from config import BASE_DIR, Config
from models import AnswerResult, AnswerSheet, Exam, Question, User, db
from modules import evaluator, report, scheme
from modules.scheme import SchemeError


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    for folder in (app.config["UPLOAD_FOLDER"], app.config["REPORT_FOLDER"],
                   os.path.join(BASE_DIR, "instance")):
        os.makedirs(folder, exist_ok=True)

    db.init_app(app)
    CSRFProtect(app)
    login_manager = LoginManager(app)
    login_manager.login_view = "login"

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def admin_required(view):
        @wraps(view)
        @login_required
        def wrapper(*args, **kwargs):
            if current_user.role != "admin":
                abort(403)
            return view(*args, **kwargs)
        return wrapper

    def owned_exam(exam_id):
        exam = db.get_or_404(Exam, exam_id)
        if current_user.role != "admin" and exam.created_by != current_user.id:
            abort(403)
        return exam

    def owned_sheet(sheet_id):
        sheet = db.get_or_404(AnswerSheet, sheet_id)
        owned_exam(sheet.exam_id)
        return sheet

    def run_evaluation(sheet, exam):
        """Read the uploaded file, evaluate every question, store the results."""
        engine = pipeline.get_ocr_engine(app)
        similarity = pipeline.get_similarity(app)
        path = os.path.join(app.config["UPLOAD_FOLDER"], sheet.stored_name)
        text = pipeline.extract_text(path, engine, app.config["PDF_RENDER_DPI"])
        sheet.raw_text = text
        for old in list(sheet.results):
            db.session.delete(old)
        for res in pipeline.evaluate_sheet_text(text, exam.questions, similarity, app.config):
            db.session.add(AnswerResult(
                sheet_id=sheet.id, question_id=res["question_id"], question_no=res["question_no"],
                extracted_text=res["extracted_text"], similarity=res["similarity"],
                keyword_score=res["keyword_score"], ai_marks=res["ai_marks"],
                final_marks=res["ai_marks"], feedback=res["feedback"]))
        sheet.status = "evaluated"
        sheet.error_message = ""
        db.session.commit()

    # ------------------------------------------------------------------
    # authentication
    # ------------------------------------------------------------------
    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            user = User.query.filter_by(username=request.form.get("username", "").strip()).first()
            if user and user.is_active and user.check_password(request.form.get("password", "")):
                login_user(user)
                nxt = request.args.get("next", "")
                return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("dashboard"))
            flash("Wrong username or password, or the account is disabled.", "error")
        return render_template("login.html")

    @app.route("/logout", methods=["POST"])
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("login"))

    # ------------------------------------------------------------------
    # exams
    # ------------------------------------------------------------------
    @app.route("/")
    @login_required
    def dashboard():
        query = Exam.query
        if current_user.role != "admin":
            query = query.filter_by(created_by=current_user.id)
        return render_template("dashboard.html", exams=query.order_by(Exam.created_at.desc()).all())

    @app.route("/exams/new", methods=["GET", "POST"])
    @login_required
    def new_exam():
        if request.method == "POST":
            title = request.form.get("title", "").strip()
            subject = request.form.get("subject", "").strip()
            file = request.files.get("scheme")
            if not title:
                flash("Please enter the exam title.", "error")
            elif not file or not file.filename.lower().endswith(".csv"):
                flash("Please upload the marking scheme as a .csv file.", "error")
            else:
                try:
                    text = file.read().decode("utf-8-sig")
                    rows = scheme.parse_scheme_csv(text)
                except UnicodeDecodeError:
                    flash("The file could not be read. Save it as CSV (UTF-8) and try again.", "error")
                except SchemeError as exc:
                    flash(str(exc), "error")
                else:
                    exam = Exam(title=title, subject=subject, created_by=current_user.id)
                    for r in rows:
                        exam.questions.append(Question(
                            number=r["number"], max_marks=r["max_marks"],
                            model_answer=r["model_answer"], keywords=r["keywords"]))
                    db.session.add(exam)
                    db.session.commit()
                    flash("Exam created with %d questions." % len(rows), "ok")
                    return redirect(url_for("exam_detail", exam_id=exam.id))
        return render_template("exam_new.html")

    @app.route("/exams/<int:exam_id>")
    @login_required
    def exam_detail(exam_id):
        return render_template("exam_detail.html", exam=owned_exam(exam_id))

    # ------------------------------------------------------------------
    # answer sheets
    # ------------------------------------------------------------------
    @app.route("/exams/<int:exam_id>/upload", methods=["GET", "POST"])
    @login_required
    def upload_sheet(exam_id):
        exam = owned_exam(exam_id)
        if request.method == "POST":
            name = request.form.get("student_name", "").strip()
            roll = request.form.get("roll_no", "").strip()
            file = request.files.get("sheet")
            ext = file.filename.rsplit(".", 1)[-1].lower() if file and "." in file.filename else ""
            error = None
            if not name or not roll:
                error = "Student name and roll number are required."
            elif not file or not file.filename:
                error = "Please choose a scanned answer sheet."
            elif ext not in app.config["ALLOWED_SHEET_EXT"]:
                error = "Only PDF, PNG, JPG or JPEG files are allowed."
            elif any(s.roll_no == roll for s in exam.sheets):
                error = f"An answer sheet for roll number {roll} was already uploaded for this exam."
            if error:
                flash(error, "error")
                return render_template("upload.html", exam=exam)

            stored = f"{uuid.uuid4().hex}.{ext}"
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], stored))
            sheet = AnswerSheet(exam_id=exam.id, student_name=name, roll_no=roll,
                                filename=secure_filename(file.filename) or stored, stored_name=stored,
                                status="processing", ocr_engine=app.config["OCR_ENGINE"])
            db.session.add(sheet)
            db.session.commit()
            try:
                run_evaluation(sheet, exam)
            except Exception as exc:  # show a readable message instead of a crash page
                db.session.rollback()
                sheet = db.session.get(AnswerSheet, sheet.id)
                sheet.status = "failed"
                sheet.error_message = str(exc)[:500]
                db.session.commit()
                flash("Evaluation failed: " + sheet.error_message, "error")
                return redirect(url_for("exam_detail", exam_id=exam.id))
            flash("Answer sheet evaluated. Please review the marks.", "ok")
            return redirect(url_for("review_sheet", sheet_id=sheet.id))
        return render_template("upload.html", exam=exam)

    @app.route("/sheets/<int:sheet_id>", methods=["GET", "POST"])
    @login_required
    def review_sheet(sheet_id):
        sheet = owned_sheet(sheet_id)
        if sheet.status == "failed":
            flash("This sheet failed to process: " + (sheet.error_message or "unknown error"), "error")
            return redirect(url_for("exam_detail", exam_id=sheet.exam_id))
        if request.method == "POST":
            action = request.form.get("action", "save")
            problems = []
            for r in sheet.results:
                r.extracted_text = request.form.get(f"text_{r.id}", r.extracted_text).strip()
                r.remarks = request.form.get(f"remarks_{r.id}", "").strip()
                if action == "recalculate":
                    res = evaluator.evaluate_answer(
                        r.extracted_text, r.question.model_answer, r.question.keyword_list,
                        r.question.max_marks, pipeline.get_similarity(app), app.config)
                    r.similarity, r.keyword_score = res["similarity"], res["keyword_score"]
                    r.ai_marks = r.final_marks = res["ai_marks"]
                    r.feedback = res["feedback"]
                    continue
                try:
                    marks = float(request.form.get(f"marks_{r.id}", r.final_marks))
                except ValueError:
                    problems.append(f"Q{r.question_no}: marks must be a number.")
                    continue
                if marks < 0 or marks > r.question.max_marks:
                    problems.append(f"Q{r.question_no}: marks must be between 0 and {r.question.max_marks:g}.")
                    continue
                if marks != r.final_marks:
                    r.reviewed = True
                r.final_marks = marks
            if problems:
                db.session.rollback()
                for p in problems:
                    flash(p, "error")
            else:
                if action == "finalize":
                    sheet.status, sheet.finalized_at = "finalized", datetime.utcnow()
                    flash("Marks finalised.", "ok")
                elif action == "recalculate":
                    sheet.status = "evaluated"
                    flash("Marks recalculated from the corrected text.", "ok")
                else:
                    sheet.status = "reviewed"
                    flash("Changes saved.", "ok")
                db.session.commit()
            return redirect(url_for("review_sheet", sheet_id=sheet.id))
        return render_template("review.html", sheet=sheet, exam=sheet.exam)

    @app.route("/sheets/<int:sheet_id>/report.pdf")
    @login_required
    def sheet_report(sheet_id):
        sheet = owned_sheet(sheet_id)
        if not sheet.results:
            abort(404)
        path = os.path.join(app.config["REPORT_FOLDER"], f"sheet_{sheet.id}.pdf")
        report.build_student_pdf(path, sheet.exam, sheet)
        return send_file(path, as_attachment=True,
                         download_name=f"{sheet.roll_no}_{sheet.exam.title}_report.pdf".replace(" ", "_"))

    @app.route("/exams/<int:exam_id>/results.csv")
    @login_required
    def exam_csv(exam_id):
        exam = owned_exam(exam_id)
        path = os.path.join(app.config["REPORT_FOLDER"], f"exam_{exam.id}.csv")
        report.build_class_csv(path, exam)
        return send_file(path, as_attachment=True, download_name=f"exam_{exam.id}_results.csv")

    # ------------------------------------------------------------------
    # administration
    # ------------------------------------------------------------------
    @app.route("/admin/users", methods=["GET", "POST"])
    @admin_required
    def manage_users():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            role = request.form.get("role", "teacher")
            if not username or len(password) < 6:
                flash("Username is required and the password needs at least 6 characters.", "error")
            elif role not in ("teacher", "admin"):
                flash("Invalid role.", "error")
            elif User.query.filter_by(username=username).first():
                flash("That username already exists.", "error")
            else:
                user = User(username=username, full_name=request.form.get("full_name", "").strip(), role=role)
                user.set_password(password)
                db.session.add(user)
                db.session.commit()
                flash(f"User '{username}' created.", "ok")
            return redirect(url_for("manage_users"))
        return render_template("users.html", users=User.query.order_by(User.username).all())

    @app.route("/admin/users/<int:user_id>/toggle", methods=["POST"])
    @admin_required
    def toggle_user(user_id):
        user = db.get_or_404(User, user_id)
        if user.id == current_user.id:
            flash("You cannot disable your own account.", "error")
        else:
            user.active_flag = not user.active_flag
            db.session.commit()
        return redirect(url_for("manage_users"))

    @app.errorhandler(403)
    def forbidden(_e):
        return render_template("error.html", code=403, message="You do not have permission to open this page."), 403

    @app.errorhandler(404)
    def not_found(_e):
        return render_template("error.html", code=404, message="Page not found."), 404

    @app.errorhandler(413)
    def too_large(_e):
        return render_template("error.html", code=413, message="The file is too large (limit 25 MB)."), 413

    # ------------------------------------------------------------------
    # command line helpers:  flask --app app init-db
    # ------------------------------------------------------------------
    @app.cli.command("init-db")
    @click.option("--admin-password", default="admin123", help="Password for the first admin user.")
    def init_db(admin_password):
        """Create tables and the first admin user."""
        db.create_all()
        if not User.query.filter_by(username="admin").first():
            admin = User(username="admin", full_name="Administrator", role="admin")
            admin.set_password(admin_password)
            db.session.add(admin)
            db.session.commit()
            click.echo("Created user 'admin'. Change the password after the first login.")
        click.echo("Database is ready.")

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
