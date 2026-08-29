from click.testing import CliRunner

from mha2gqa.cli import cli


def test_convert_cli_end_to_end(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, [
        "convert",
        "--model-id", "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "--output-dir", str(tmp_path / "out"),
        "--kv-groups", "1",
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "config.json").exists()