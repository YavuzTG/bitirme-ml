# Auth Widgets ve Yönetim - Modern UI
from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QMessageBox, QTabWidget, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QIcon, QColor, QPixmap
import re

class ModernLineEdit(QLineEdit):
    """Modern stillendirilmiş QLineEdit"""
    def __init__(self, placeholder=""):
        super().__init__()
        self.setPlaceholderText(placeholder)
        self.setMinimumHeight(45)
        self.setStyleSheet("""
            QLineEdit {
                background: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 10px 14px;
                font-size: 13px;
                font-family: 'Segoe UI';
                color: #1f1f1f;
            }
            QLineEdit:focus {
                border: 2px solid #3b82f6;
                background: #f8fbff;
            }
            QLineEdit::placeholder {
                color: #999999;
            }
        """)

class ModernButton(QPushButton):
    """Modern stillendirilmiş QPushButton"""
    def __init__(self, text, is_primary=True):
        super().__init__(text)
        self.setMinimumHeight(45)
        self.setFont(QFont("Segoe UI", 11, QFont.Bold))
        
        if is_primary:
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(
                        x1: 0, y1: 0, x2: 1, y2: 1,
                        stop: 0 #3b82f6, stop: 1 #2563eb
                    );
                    color: #ffffff;
                    border: none;
                    border-radius: 10px;
                    font-weight: bold;
                    padding: 10px;
                }
                QPushButton:hover {
                    background: qlineargradient(
                        x1: 0, y1: 0, x2: 1, y2: 1,
                        stop: 0 #2563eb, stop: 1 #1d4ed8
                    );
                }
                QPushButton:pressed {
                    background: qlineargradient(
                        x1: 0, y1: 0, x2: 1, y2: 1,
                        stop: 0 #1d4ed8, stop: 1 #1e40af
                    );
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background: #f3f4f6;
                    color: #374151;
                    border: none;
                    border-radius: 10px;
                    font-weight: bold;
                    padding: 10px;
                }
                QPushButton:hover {
                    background: #e5e7eb;
                }
                QPushButton:pressed {
                    background: #d1d5db;
                }
            """)

class LoginDialog(QDialog):
    """Modern Firebase giriş/kayıt dialog"""
    login_success = pyqtSignal(str, bool)  # email, is_admin
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BEED — Giriş")
        self.setMinimumSize(500, 620)
        self.setMaximumSize(500, 620)
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        self.setModal(True)
        
        self.auth = None
        try:
            from firebase_config import auth as firebase_auth
            self.auth = firebase_auth
        except ImportError:
            pass
            
        self._build_ui()
        self._apply_style()
    
    def _apply_style(self):
        """Modern stylesheet uygula"""
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #f8fafc, stop: 1 #e2e8f0
                );
            }
        """)
    
    def _build_ui(self):
        """Modern UI oluştur"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)
        
        # Header: Logo ve Başlık
        header_layout = QVBoxLayout()
        header_layout.setSpacing(10)
        header_layout.setAlignment(Qt.AlignCenter)
        
        title = QLabel("BEED")
        title.setFont(QFont("Segoe UI", 32, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1e3a8a;")
        
        subtitle = QLabel("Sinyal Sınıflandırma Sistemi")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #64748b;")
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        main_layout.addLayout(header_layout)
        
        # Separator
        sep1 = QFrame()
        sep1.setStyleSheet("background: #cbd5e1; margin: 10px 0px;")
        sep1.setFixedHeight(1)
        main_layout.addWidget(sep1)
        
        # Tab Widget
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
            }
            QTabBar::tab {
                background: #f1f5f9;
                color: #475569;
                padding: 12px 24px;
                margin: 0px 2px;
                border-radius: 8px 8px 0px 0px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #3b82f6;
                color: #ffffff;
            }
        """)
        
        # Giriş Sekmesi
        login_tab = self._build_login_tab()
        tabs.addTab(login_tab, "Giriş")
        
        # Kayıt Sekmesi
        register_tab = self._build_register_tab()
        tabs.addTab(register_tab, "Kayıt Ol")
        
        main_layout.addWidget(tabs)
        
        # Footer
        footer = QLabel("Güvenli Firebase Authentication ile korunuyor")
        footer.setFont(QFont("Segoe UI", 9))
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #94a3b8;")
        main_layout.addWidget(footer)
        
        self.setLayout(main_layout)
    
    def _build_login_tab(self):
        """Giriş sekmesi"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(14)
        
        # Email
        email_lbl = QLabel("Email Adresi")
        email_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        email_lbl.setStyleSheet("color: #334155;")
        self.login_email = ModernLineEdit("example@gmail.com")
        layout.addWidget(email_lbl)
        layout.addWidget(self.login_email)
        
        # Şifre
        pass_lbl = QLabel("Şifre")
        pass_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        pass_lbl.setStyleSheet("color: #334155;")
        self.login_password = ModernLineEdit("••••••••")
        self.login_password.setEchoMode(QLineEdit.Password)
        layout.addWidget(pass_lbl)
        layout.addWidget(self.login_password)
        
        layout.addSpacing(10)
        
        # Giriş Butonu
        login_btn = ModernButton("Giriş Yap", is_primary=True)
        login_btn.clicked.connect(self._handle_login)
        layout.addWidget(login_btn)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def _build_register_tab(self):
        """Kayıt sekmesi"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(14)
        
        # Email
        email_lbl = QLabel("Email Adresi")
        email_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        email_lbl.setStyleSheet("color: #334155;")
        self.reg_email = ModernLineEdit("example@gmail.com")
        layout.addWidget(email_lbl)
        layout.addWidget(self.reg_email)
        
        # Şifre
        pass_lbl = QLabel("Şifre (min 6 karakter)")
        pass_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        pass_lbl.setStyleSheet("color: #334155;")
        self.reg_password = ModernLineEdit("••••••••")
        self.reg_password.setEchoMode(QLineEdit.Password)
        layout.addWidget(pass_lbl)
        layout.addWidget(self.reg_password)
        
        # Şifre Tekrar
        pass_confirm_lbl = QLabel("Şifre Tekrar")
        pass_confirm_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        pass_confirm_lbl.setStyleSheet("color: #334155;")
        self.reg_password_confirm = ModernLineEdit("••••••••")
        self.reg_password_confirm.setEchoMode(QLineEdit.Password)
        layout.addWidget(pass_confirm_lbl)
        layout.addWidget(self.reg_password_confirm)
        
        layout.addSpacing(10)
        
        # Kayıt Butonu
        register_btn = ModernButton("Kayıt Ol", is_primary=True)
        register_btn.clicked.connect(self._handle_register)
        layout.addWidget(register_btn)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def _handle_login(self):
        """Firebase ile giriş yap"""
        if not self.auth:
            QMessageBox.critical(
                self, "Hata", 
                "Firebase baglantisi kurulamadi.\nLutfen firebase_config.py kontrol et!"
            )
            return
        
        email = self.login_email.text().strip()
        password = self.login_password.text()
        
        if not email or not password:
            QMessageBox.warning(self, "Uyarı", "Email ve sifre bos olamaz!")
            return
        
        try:
            user = self.auth.sign_in_with_email_and_password(email, password)
            is_admin = self._check_if_admin(email)
            self.login_success.emit(email, is_admin)
            self.accept()
        except Exception as e:
            error_msg = str(e)
            if "EMAIL_NOT_FOUND" in error_msg or "INVALID_PASSWORD" in error_msg:
                QMessageBox.warning(self, "Hata", "Email veya sifre yanlis!")
            else:
                QMessageBox.critical(self, "Hata", f"Giris basarisiz:\n{error_msg}")
    
    def _handle_register(self):
        """Firebase ile kayıt ol"""
        if not self.auth:
            QMessageBox.critical(self, "Hata", "Firebase baglantisi kurulamadi!")
            return
        
        email = self.reg_email.text().strip()
        password = self.reg_password.text()
        password_confirm = self.reg_password_confirm.text()
        
        # Doğrulama
        if not email or not password:
            QMessageBox.warning(self, "Uyarı", "Email ve sifre bos olamaz!")
            return
        
        if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
            QMessageBox.warning(self, "Uyarı", "Gecerli bir email adresi gir!")
            return
        
        if len(password) < 6:
            QMessageBox.warning(self, "Uyarı", "Sifre minimum 6 karakter olmali!")
            return
        
        if password != password_confirm:
            QMessageBox.warning(self, "Uyarı", "Sifreler eslesmiyor!")
            return
        
        try:
            self.auth.create_user_with_email_and_password(email, password)
            user = self.auth.sign_in_with_email_and_password(email, password)
            is_admin = self._check_if_admin(email)
            self.login_success.emit(email, is_admin)
            QMessageBox.information(self, "Basarili", "Kayit ve giris basarili.")
            self.accept()
        except Exception as e:
            error_msg = str(e)
            if "EMAIL_EXISTS" in error_msg:
                QMessageBox.warning(self, "Hata", "Bu email zaten kayitli!")
            elif "WEAK_PASSWORD" in error_msg:
                QMessageBox.warning(self, "Hata", "Sifre cok zayif!")
            else:
                QMessageBox.critical(self, "Hata", f"Kayit basarisiz:\n{error_msg}")
    
    def _check_if_admin(self, email):
        """Email'in admin olup olmadığını kontrol et"""
        ADMIN_EMAILS = ["yavuzturker@icloud.com"]
        return email.lower() in ADMIN_EMAILS
