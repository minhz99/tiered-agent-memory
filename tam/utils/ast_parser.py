import ast
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ASTAnalyzer:
    """
    Phân tích cấu trúc mã nguồn Python để trích xuất metadata ngữ cảnh.
    Giúp TAM hiểu code sâu hơn chỉ là văn bản thuần túy.
    """
    
    @staticmethod
    def extract_metadata(code: str) -> Dict[str, Any]:
        """Trích xuất class, function names và docstrings từ code."""
        metadata = {
            "functions": [],
            "classes": [],
            "docstrings": [],
            "imports": [],
            "is_code": False
        }
        
        try:
            tree = ast.parse(code)
            metadata["is_code"] = True
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metadata["functions"].append(node.name)
                    doc = ast.get_docstring(node)
                    if doc:
                        metadata["docstrings"].append(doc)
                        
                elif isinstance(node, ast.ClassDef):
                    metadata["classes"].append(node.name)
                    doc = ast.get_docstring(node)
                    if doc:
                        metadata["docstrings"].append(doc)
                        
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            metadata["imports"].append(alias.name)
                    else:
                        metadata["imports"].append(node.module)
            
            # Lấy docstring tổng của file/đoạn code
            module_doc = ast.get_docstring(tree)
            if module_doc:
                metadata["docstrings"].insert(0, module_doc)
                
        except SyntaxError:
            # Nếu không parse được bằng AST thì có thể không phải là code Python hoàn chỉnh
            metadata["is_code"] = False
            
        return metadata

    @staticmethod
    def get_context_summary(metadata: Dict[str, Any]) -> str:
        """Tạo một bản tóm tắt ngắn gọn từ metadata để hỗ trợ embedding."""
        if not metadata["is_code"]:
            return ""
            
        summary = []
        if metadata["classes"]:
            summary.append(f"Classes: {', '.join(metadata['classes'])}")
        if metadata["functions"]:
            summary.append(f"Functions: {', '.join(metadata['functions'])}")
        if metadata["imports"]:
            summary.append(f"Dependencies: {', '.join(metadata['imports'][:5])}")
            
        return " | ".join(summary)
