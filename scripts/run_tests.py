"""
测试运行脚本

运行所有测试并生成覆盖率报告
"""
import subprocess
import sys
import os
from pathlib import Path

def run_backend_tests():
    """运行后端测试"""
    print("=" * 60)
    print("运行后端测试...")
    print("=" * 60)
    
    backend_dir = Path(__file__).parent.parent / "backend"
    os.chdir(backend_dir)
    
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/",
                "-v",
                "--tb=short",
                "--cov=src",
                "--cov-report=html:htmlcov",
                "--cov-report=term",
            ],
            check=False
        )
        
        if result.returncode == 0:
            print("\n✅ 后端测试通过")
        else:
            print("\n❌ 后端测试失败")
        
        return result.returncode
    except Exception as e:
        print(f"\n❌ 运行后端测试时出错: {e}")
        return 1

def run_frontend_tests():
    """运行前端测试"""
    print("\n" + "=" * 60)
    print("运行前端测试...")
    print("=" * 60)
    
    frontend_dir = Path(__file__).parent.parent / "frontend"
    os.chdir(frontend_dir)
    
    try:
        result = subprocess.run(
            ["npm", "test", "--", "--run"],
            check=False
        )
        
        if result.returncode == 0:
            print("\n✅ 前端测试通过")
        else:
            print("\n❌ 前端测试失败")
        
        return result.returncode
    except Exception as e:
        print(f"\n❌ 运行前端测试时出错: {e}")
        return 1

def generate_report():
    """生成测试报告"""
    print("\n" + "=" * 60)
    print("测试报告")
    print("=" * 60)
    
    backend_dir = Path(__file__).parent.parent / "backend"
    frontend_dir = Path(__file__).parent.parent / "frontend"
    
    print("\n📊 后端测试覆盖率报告:")
    print(f"   HTML报告: {backend_dir}/htmlcov/index.html")
    
    print("\n📊 前端测试覆盖率报告:")
    print(f"   运行 'npm run test:coverage' 查看详细报告")

def main():
    """主函数"""
    print("🧪 GraphRAG 测试套件")
    print("=" * 60)
    
    backend_result = run_backend_tests()
    frontend_result = run_frontend_tests()
    
    generate_report()
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if backend_result == 0 and frontend_result == 0:
        print("✅ 所有测试通过")
        return 0
    else:
        print("❌ 部分测试失败")
        if backend_result != 0:
            print("   - 后端测试失败")
        if frontend_result != 0:
            print("   - 前端测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
