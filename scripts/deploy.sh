#!/bin/bash

# 退出脚本如果发生任何错误
set -e

export HOME_DIR="$(dirname "$(dirname $(readlink -f $0))")"
echo "HOME_DIR: ${HOME_DIR}"
cd ${HOME_DIR}

# 定义源路径和目标路径
BUILD_DIR="build"
DEPLOY_DIR="deploy"
LIB_DEST="$DEPLOY_DIR/lib"
BIN_DEST="$DEPLOY_DIR/bin"

echo "=== 开始准备部署目录 ==="

# 1. 创建干净的部署目录结构
rm -rf "$DEPLOY_DIR" "$DEPLOY_DIR.tar.gz"
mkdir -p "$LIB_DEST"
mkdir -p "$BIN_DEST"

# 2. 拷贝动态链接库 (.so)
# 使用 -d 选项（或 -a）以保持软链接本身，而不是拷贝软链接指向的源文件
echo "正在拷贝动态链接库..."

# 拷贝 build/examples/llm/ 下的 so
if [ -d "$BUILD_DIR/examples/llm" ]; then
    find "$BUILD_DIR/examples/llm" -maxdepth 1 -name "*.so*" -exec cp -d {} "$LIB_DEST/" \;
fi

# 拷贝 build/ 下的 so
find "$BUILD_DIR" -maxdepth 1 -name "*.so*" -exec cp -d {} "$LIB_DEST/" \;

# 3. 拷贝可执行程序
# 通过 find 查找类型为文件 (-type f) 且具有执行权限 (-executable) 的文件
echo "正在拷贝可执行程序..."

copy_executables() {
    local src_dir=$1
    if [ -d "$src_dir" ]; then
        # 排除 .so 文件，只拷贝真正的二进制可执行程序
        find "$src_dir" -maxdepth 1 -type f -executable ! -name "*.so*" -exec cp {} "$BIN_DEST/" \;
    fi
}

copy_executables "$BUILD_DIR/examples/llm"
copy_executables "$BUILD_DIR/examples/multimodal"

# 4. 打包压缩
echo "正在打包压缩为 $DEPLOY_DIR.tar.gz ..."
# tar -zcvf 中的 -h 默认会解开软链接，这里我们**不使用 -h**，以保持生成的 tar.gz 内依然是软链接
tar -zcvf "$DEPLOY_DIR.tar.gz" "$DEPLOY_DIR"

echo "=== 部署打包完成！ ==="
ls -lh "$DEPLOY_DIR.tar.gz"
