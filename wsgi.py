"""Entry point for hosting (gunicorn wsgi:app). Creates tables and the first admin."""
import os
from app import create_app
from models import db, User

app = create_app()

with app.app_context():
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["REPORT_FOLDER"], exist_ok=True)
    db.create_all()
    if not User.query.filter_by(username="admin").first():
        admin = User(username="admin", full_name="Administrator", role="admin")
        admin.set_password(os.environ.get("ADMIN_PASSWORD", "admin123"))
        db.session.add(admin)
        db.session.commit()
