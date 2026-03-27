from app.utils.validators import URLValidator, CodeValidator

def test_url_validator():
    assert URLValidator.is_valid_url("https://github.com") == True
    assert URLValidator.is_valid_url("http://localhost:8000") == False
    assert URLValidator.is_valid_url("http://192.168.1.1") == False

def test_code_validator():
    is_safe, warning = CodeValidator.is_safe_code("print('hello world')")
    assert is_safe == True
    
    is_safe, warning = CodeValidator.is_safe_code("import os\nos.system('rm -rf /')")
    assert is_safe == False
    assert "dangerous operation" in warning
