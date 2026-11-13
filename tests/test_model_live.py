"""线上模型的一致性套件（开发文档 11.3、12.2；AC-11-4）。

按部署环境变量（EAER_GEN_*、EAER_EMBED_*、EAER_RERANK_*，见 eaer/adapters/online.py）连接真实服务；
未设置 EAER_GEN_BASE_URL 时整个文件跳过。没有配置的服务，相应用例显式跳过并说明原因：
Fast 模式需要生成、Embedding、重排三项（12.1），缺任何一项都不能上线。会产生少量调用费用。
"""
from __future__ import annotations

import os

import pytest

from eaer.adapters.conformance import ModelClientContract
from eaer.adapters.online import OnlineModelClient

pytestmark = pytest.mark.skipif(not os.environ.get("EAER_GEN_BASE_URL"), reason="未设置 EAER_GEN_BASE_URL")


class TestOnlineModel(ModelClientContract):
    @pytest.fixture(scope="class")
    def model(self):
        return OnlineModelClient.from_env()

    def test_capabilities_are_well_formed(self, model):
        if model.emb is None:
            pytest.skip("未配置 Embedding 服务：一致性要求 embedding_dims > 0（Fast 必需，12.1）")
        super().test_capabilities_are_well_formed(model)

    def test_embed_shapes_and_stability(self, model):
        if model.emb is None:
            pytest.skip("未配置 Embedding 服务")
        super().test_embed_shapes_and_stability(model)

    def test_rerank_scores_every_passage(self, model):
        if model.rr is None:
            pytest.skip("未配置重排服务")
        super().test_rerank_scores_every_passage(model)


def test_fast_requirements_are_reported():
    missing = OnlineModelClient.from_env().missing_for_fast()
    assert "生成" not in missing                       # 生成服务已配置
