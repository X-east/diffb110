"""
防回归测试：确保活跃代码中不出现旧 RLPF/verl_diffusion 依赖。

检查范围：ddpo/ configs/ launch/ 目录下所有 .py 文件。
检查项：
    1. from verl_diffusion / import verl_diffusion
    2. RLPF-verl_diffusion 路径字符串
    3. worker.actor.edm_actor 引用
    4. from verl（独立 verl 模块导入）
"""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECK_DIRS = ["ddpo", "configs", "launch"]

FORBIDDEN_IMPORTS = [
    ("from verl_diffusion", "from verl_diffusion ..."),
    ("import verl_diffusion", "import verl_diffusion ..."),
    ("from verl ", "from verl ... (verl 独立模块)"),
    ("import verl", "import verl ... (verl 独立模块)"),
]

FORBIDDEN_STRINGS = [
    "RLPF-verl_diffusion",
    "worker.actor.edm_actor",
]


def _collect_py_files(root: Path, subdirs: list[str]) -> list[Path]:
    """收集指定子目录下所有 .py 文件。"""
    files = []
    for sub in subdirs:
        target = root / sub
        if not target.is_dir():
            continue
        files.extend(target.rglob("*.py"))
    return sorted(files)


class TestNoRLPFRegression(unittest.TestCase):
    """验证活跃代码中无 RLPF/verl_diffusion 残留引用。"""

    @classmethod
    def setUpClass(cls):
        cls.py_files = _collect_py_files(PROJECT_ROOT, CHECK_DIRS)

    def _read_file_lines(self, path: Path) -> list[str]:
        """安全读取文件，跳过编码错误。"""
        try:
            return path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []

    def test_no_forbidden_imports(self):
        """检查无 verl_diffusion / verl 导入。"""
        violations = []
        for f in self.py_files:
            lines = self._read_file_lines(f)
            for line_no, line in enumerate(lines, 1):
                for pattern, desc in FORBIDDEN_IMPORTS:
                    if pattern in line and not line.strip().startswith("#"):
                        violations.append(f"  {f.relative_to(PROJECT_ROOT)}:{line_no}: {desc}")

        if violations:
            self.fail(
                f"发现 {len(violations)} 处禁止的导入引用:\n" + "\n".join(violations)
            )

    def test_no_forbidden_path_strings(self):
        """检查无 RLPF-verl_diffusion 路径字符串 / edm_actor 引用。"""
        violations = []
        for f in self.py_files:
            text = f.read_text(encoding="utf-8", errors="replace")
            for pattern in FORBIDDEN_STRINGS:
                if pattern in text:
                    # 定位行号
                    for line_no, line in enumerate(text.splitlines(), 1):
                        if pattern in line and not line.strip().startswith("#"):
                            violations.append(
                                f"  {f.relative_to(PROJECT_ROOT)}:{line_no}: {pattern}"
                            )
                            break  # 每个文件每种 pattern 只报一次

        if violations:
            self.fail(
                f"发现 {len(violations)} 处禁止的路径/模块引用:\n" + "\n".join(violations)
            )

    def test_no_forbidden_ast_imports(self):
        """AST 级别检查：确保没有 import 语句引用 verl_diffusion。"""
        violations = []
        for f in self.py_files:
            try:
                tree = ast.parse(f.read_text(encoding="utf-8"))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if "verl_diffusion" in alias.name or alias.name == "verl":
                            violations.append(
                                f"  {f.relative_to(PROJECT_ROOT)}:{node.lineno}: import {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module and ("verl_diffusion" in node.module or node.module == "verl"):
                        violations.append(
                            f"  {f.relative_to(PROJECT_ROOT)}:{node.lineno}: from {node.module} import ..."
                        )

        if violations:
            self.fail(
                f"发现 {len(violations)} 处 AST 级别的禁止导入:\n" + "\n".join(violations)
            )

    def test_check_dirs_exist(self):
        """确保检查目录存在（防止配置漂移导致测试失效）。"""
        for sub in CHECK_DIRS:
            target = PROJECT_ROOT / sub
            self.assertTrue(
                target.is_dir(),
                f"检查目录不存在: {target}，可能路径配置漂移",
            )

    def test_check_some_files_found(self):
        """确保至少找到了文件（防止空跑）。"""
        self.assertGreater(
            len(self.py_files), 0,
            f"未在 {CHECK_DIRS} 中找到任何 .py 文件",
        )


if __name__ == "__main__":
    unittest.main()
