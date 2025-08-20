# Changelog

All notable changes to the hydroutils project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.0.14] - 2025-08-19

### Added
- 完整的项目文档结构，包括API参考、使用指南和示例
- 新增水文统计分析模块 (`hydro_stat`)
  - 支持NSE、KGE、RMSE等多种评价指标
  - 洪水事件提取和分析功能
  - 流量持续曲线分析
- 时间序列处理模块 (`hydro_time`)
  - 时间间隔检测和验证
  - 单位转换功能
- 可视化工具模块 (`hydro_plot`)
  - 水文数据专用绘图函数
  - 模型评价可视化工具
  - 发布级别图表输出
- AWS S3集成模块 (`hydro_s3`)
  - 支持大规模水文数据云存储
  - 批量数据上传下载
- 日志工具模块 (`hydro_log`)
  - 专业的水文分析日志记录
  - 性能监控和错误追踪

### Changed
- 重构了项目结构，优化模块组织
- 改进了数据处理流程，提高计算效率
- 更新了所有依赖包的版本要求
- 统一了代码风格和文档格式

### Fixed
- 修复了统计计算中的NaN值处理问题
- 解决了时间序列对齐的bug
- 修正了单位转换的精度问题
- 优化了内存使用，解决了大数据处理时的内存溢出

### Deprecated
- 移除了过时的数据格式支持
- 废弃了部分不推荐使用的函数接口

## [v0.0.13] - 2025-07-15

### Added
- 初始版本发布
- 基础的水文统计功能
- 简单的数据处理工具

### Changed
- 基础功能实现和测试

## [Unreleased]

### Planned
- 增加机器学习模块支持
- 添加更多水文模型评价指标
- 改进数据可视化功能
- 优化大规模数据处理性能
- 添加更多单元测试和集成测试

---

## Version Number Guide

- **MAJOR** version (x.0.0) - 不兼容的API修改
- **MINOR** version (0.x.0) - 向后兼容的功能性新增
- **PATCH** version (0.0.x) - 向后兼容的问题修复

## Links
- [v0.0.14]: https://github.com/zhuanglaihong/hydroutils/releases/tag/v0.0.14
- [v0.0.13]: https://github.com/zhuanglaihong/hydroutils/releases/tag/v0.0.13
