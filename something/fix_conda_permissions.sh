#!/bin/bash

# ============================================================================
# Conda 权限问题 - 快速诊断和修复脚本
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Conda 权限问题 - 诊断和修复工具                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# 诊断
# ============================================================================
echo -e "${YELLOW}[1/3] 诊断权限问题...${NC}"
echo ""

MINIFORGE_DIR="/home/md666/miniforge3"
ENVS_DIR="$MINIFORGE_DIR/envs"
CURRENT_USER=$(whoami)

echo "Miniforge 目录: $MINIFORGE_DIR"
echo "当前用户: $CURRENT_USER"
echo ""

# 检查所有者
OWNER=$(ls -ld "$ENVS_DIR" | awk '{print $3}')
echo "Envs 目录所有者: $OWNER"

if [ "$OWNER" != "$CURRENT_USER" ]; then
    echo -e "${RED}✗ 权限问题: 目录由 $OWNER 拥有，但你是 $CURRENT_USER${NC}"
    NEEDS_FIX=true
else
    echo -e "${GREEN}✓ 权限正确${NC}"
    NEEDS_FIX=false
fi

# 测试写入
echo ""
echo "测试写入权限..."
if touch "$ENVS_DIR/.test" 2>/dev/null; then
    rm "$ENVS_DIR/.test"
    echo -e "${GREEN}✓ 可以写入${NC}"
    CAN_WRITE=true
else
    echo -e "${RED}✗ 无法写入${NC}"
    CAN_WRITE=false
fi

# ============================================================================
# 修复选项
# ============================================================================
echo ""
echo -e "${YELLOW}[2/3] 修复选项...${NC}"
echo ""

if [ "$NEEDS_FIX" = true ] || [ "$CAN_WRITE" = false ]; then
    echo "检测到权限问题。选择修复方案:"
    echo ""
    echo "1) 修复权限 (需要 sudo)"
    echo "2) 使用本地项目环境 (无需 sudo)"
    echo "3) 跳过 (稍后手动修复)"
    echo ""
    read -p "选择 (1-3): " choice
    
    case $choice in
        1)
            echo ""
            echo -e "${YELLOW}修复权限...${NC}"
            if sudo chown -R "$CURRENT_USER:$CURRENT_USER" "$MINIFORGE_DIR" 2>/dev/null; then
                echo -e "${GREEN}✓ 权限已修复${NC}"
                NEEDS_FIX=false
            else
                echo -e "${RED}✗ 需要 sudo 权限${NC}"
            fi
            ;;
        2)
            echo ""
            echo -e "${YELLOW}设置本地项目环境...${NC}"
            PROJECT_DIR="/home/md666/data/disk6/DiffB110"
            LOCAL_ENVS="$PROJECT_DIR/.envs"
            
            mkdir -p "$LOCAL_ENVS"
            echo -e "${GREEN}✓ 创建本地环境目录: $LOCAL_ENVS${NC}"
            
            # 配置 .condarc
            cat >> ~/.condarc << 'CONDARC'

# 本地项目环境
envs_dirs:
  - /home/md666/data/disk6/DiffB110/.envs
CONDARC
            
            echo -e "${GREEN}✓ 已配置 ~/.condarc${NC}"
            echo ""
            echo "现在可以创建本地环境:"
            echo "  conda create -p $LOCAL_ENVS/dif2 python=3.10 -y"
            ;;
        3)
            echo "跳过修复"
            ;;
    esac
else
    echo -e "${GREEN}✓ 没有检测到权限问题${NC}"
fi

# ============================================================================
# 验证
# ============================================================================
echo ""
echo -e "${YELLOW}[3/3] 验证修复...${NC}"
echo ""

if touch "$ENVS_DIR/.test" 2>/dev/null; then
    rm "$ENVS_DIR/.test"
    echo -e "${GREEN}✓ 权限问题已解决！${NC}"
    echo ""
    echo "现在可以运行:"
    echo "  conda create -n dif2 python=3.10 -y"
else
    echo -e "${YELLOW}⚠ 权限问题仍然存在${NC}"
    echo ""
    echo "请参考: PERMISSION_ISSUES_GUIDE.md"
fi

echo ""
