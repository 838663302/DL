"""单文件：把训练产出的 .pth 权重上传到 Kaggle 数据集，永久保存。

这个文件不依赖项目里任何其他文件，可以直接放进 Kaggle Notebook 单独运行。
也可以在本机单独测试（非 Kaggle 环境会自动跳过）。

用法一（Kaggle Notebook 里直接跑）：
    1. 把这个文件上传到 Notebook 的 /kaggle/working
    2. 改下面的 两个配置 为你自己的值
    3. 运行：!python /kaggle/working/upload_to_kaggle.py
    或在 cell 里：import upload_to_kaggle; upload_to_kaggle.main()

用法二（训练脚本内调用，随时手动上传）：
    import upload_to_kaggle
    upload_to_kaggle.main()   # 会把下方 SOURCE_DIR 里所有 .pth 上传
"""

import glob
import os
import shutil

# ===================== 配置区（改成你自己的） =====================
# 目标数据集，格式必须是 {你的Kaggle用户名}/{数据集slug}。
# 第一次上传会自动创建该数据集，之后再传会追加为新版本。
KAGGLE_DATASET_HANDLE = "xiaonanhaiaichixigua/model-checkpoints"

# 要上传的 .pth 所在目录（Kaggle 训练脚本把模型存哪就填哪）
SOURCE_DIR = "/kaggle/working"

# 每次上传的版本说明（可选）
VERSION_NOTES = "模型权重"
# ================================================================


def upload_checkpoints(handle=KAGGLE_DATASET_HANDLE, source_dir=SOURCE_DIR,
                       version_notes=VERSION_NOTES):
    """把 source_dir 下所有 .pth 上传到 handle 指定的 Kaggle 数据集。

    返回 True = 已上传；False = 被跳过（非 Kaggle / 无文件 / 出错）。
    任何失败都只打印警告，不会抛异常。
    """
    # 1) 只在 Kaggle 环境执行
    if not os.path.exists("/kaggle/input"):
        print("[UPLOAD] 非 Kaggle 环境，跳过上传")
        return False

    files = sorted(glob.glob(os.path.join(str(source_dir), "*.pth")))
    if not files:
        print(f"[UPLOAD] {source_dir} 下没有 .pth 文件，跳过上传")
        return False

    # 2) 确保 kagglehub 可用
    try:
        import kagglehub
    except ImportError:
        print("[UPLOAD] kagglehub 未安装，自动安装 ...")
        try:
            import subprocess
            import sys
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "-U", "kagglehub"])
            import kagglehub
        except Exception as e:
            print(f"[UPLOAD] kagglehub 安装失败，跳过上传: {e}")
            return False

    # 3) 拷到独立中转目录，只上传 .pth
    staging = "/kaggle/working/upload_model"
    os.makedirs(staging, exist_ok=True)
    for f in files:
        shutil.copy(f, os.path.join(staging, os.path.basename(f)))
    names = [os.path.basename(f) for f in files]
    print(f"[UPLOAD] 待上传: {names} -> 数据集 {handle}")

    # 4) 上传（创建新数据集或追加新版本）
    try:
        kagglehub.dataset_upload(handle, staging, version_notes=version_notes)
        print("[UPLOAD] 上传完成，权重已永久保存到 Kaggle 数据集")
        return True
    except Exception as e:
        print(f"[UPLOAD] 上传失败（不影响本地文件）: {e}")
        return False


def main():
    ok = upload_checkpoints()
    print("[UPLOAD] 结果:", "成功" if ok else "未执行/失败")


if __name__ == "__main__":
    main()
