import os
import sys
import asyncio
import time
from mini_harness.infra.tools import create_default_tools, WORKSPACE_ROOT


# =============== 新增兼容windows和Linux双系统测试 =========================
def is_windows():
    return sys.platform == "win32"


async def demo():
    print("=== 准备沙箱环境 ===")
    # 在 workspace 中创建测试文件 a.txt
    test_file = os.path.join(WORKSPACE_ROOT, "a.txt")
    with open(test_file, "w") as f:
        f.write("Hello, Mini Harness!\nThis is a test file.")

    registry = create_default_tools()

    # ---------- 并发测试（read + shell） ----------
    async def read_task():
        print(f"[{time.strftime('%H:%M:%S')}] >>> 开始读取 a.txt")
        await asyncio.sleep(0.5)  # 模拟轻微延迟，帮助观察并发

        try:
            result = await registry.execute_with_retry("read_file", path="a.txt")
            print(f"[{time.strftime('%H:%M:%S')}] \
                  读取成功: content={result['content'][:50]}, \
                  size={result['size']}")
            
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 读取失败: {e}")
    
    async def shell_task():
        # 使用跨平台命令：Windows 用 dir，Linux 用 ls -la
        cmd = "dir" if is_windows() else "ls -la"
        print(f"[{time.strftime('%H:%M:%S')}] >>> 开始执行 {cmd}")
        await asyncio.sleep(0.5)
        try:
            result = await registry.execute_with_retry(
                "execute_command", 
                command=cmd, 
                timeout=5.0
            )

            print(f"[{time.strftime('%H:%M:%S')}] \
                  命令执行成功: return_code={result['return_code']}")
            print(f"    stdout: {result['stdout'][:100]}")
            if result['stderr']:
                print(f"    stderr: {result['stderr'][:100]}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 命令执行失败: {e}")
    # 并发执行
    await asyncio.gather(read_task(), shell_task())

    # 测试超时：执行一个长时间命令，预期超时    
    print("\n=== 测试超时（10秒命令，超时3秒） ===")
    if is_windows():
        cmd = "ping -n 10 127.0.0.1"   # 大约 9 秒完成
    else:
        cmd = "sleep 10"
    
    try:
        result = await registry.execute_with_retry(
            "execute_command", 
            command=cmd, 
            timeout=3.0
        )
        print(f"超时测试结果: timed_out={result['timed_out']}, \
              return_code={result['return_code']}")
        if result['stderr']:
            print(f"    stderr: {result['stderr'][:100]}")
    except Exception as e:
        print(f"超时测试异常: {e}")

    # 测试重试：故意读取不存在的文件（read_file 内部会抛异常）
    print("\n=== 测试重试（读取不存在的 b.txt） ===")
    
    # 为了触发重试，需要让工具函数在失败时抛出异常。
    # 但默认 read_file 返回 error 字段而不抛异常。
    # 因此：在注册时包装函数，检查返回值若有 error 则抛异常。
    # 为了演示，临时修改 registry 中的 func 或直接调用 sandbox。
    # 这里直接调用 sandbox 的方法演示重试效果。
    # 实际使用时，可以在注册时增加异常抛出逻辑。
    # 下面演示手动调用 execute_with_retry 包装一个会失败的函数。
    async def failing_read():
        """
        为了触发重试，需要让工具函数在失败时抛出异常。
        但默认 read_file 返回 error 字段而不抛异常。
        因此：在注册时包装函数，检查返回值若有 error 则抛异常。
        为了演示，临时修改 registry 中的 func 或直接调用 sandbox。
        这里直接调用 sandbox 的方法演示重试效果。
        实际使用时，可以在注册时增加异常抛出逻辑。
        下面演示手动调用 execute_with_retry 包装一个会失败的函数。
        
        """
        res = await registry.sandbox.read_file("not_exist.txt")
        if res.get("error"):
            raise RuntimeError(res["error"])
        return res
    
    # 临时注册一个工具用于测试重试
    registry.register(
        name="test_fail",
        description="测试重试",
        parameters={"type": "object", "properties": {}},
        func=failing_read
    )
    try:
        await registry.execute_with_retry("test_fail")

    except Exception as e:
        print(f"重试后最终失败: {e}")

    #================= 明确超时测试====================
    
    print("=== 测试真正超时（sleep 60，超时3秒） ===")
    try:
        # Windows 用 timeout /t 60，Linux 用 sleep 60
        cmd = "ping -n 60 127.0.0.1 > nul" if sys.platform == "win32" else "sleep 60"
        result = await registry.execute_with_retry(
            "execute_command", 
            command=cmd,
            timeout=3.0  # 确保这里传入了 timeout
        )
         # 注意：execute_with_retry 不会抛出 TimeoutError，而是返回结果字典带 timed_out=True
        if result.get('timed_out'):
            print("✅ 超时生效！timed_out=True")
        else:
            print(f"⚠️ 命令提前结束？return_code={result['return_code']}, stdout={result['stdout'][:50]}")
    except Exception as e:
        print(f"❌ 执行异常: {e}")
        if e.__cause__:  # 👈 新增
            print(f"原因: {e.__cause__}")
    
    print("\n演示结束。")

if __name__ == "__main__":
    asyncio.run(demo())