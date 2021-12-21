from typing import Sequence, Literal, Union, NamedTuple

import vh3


class Python3DeepLearningOutput(NamedTuple):
    data: vh3.Binary
    state: vh3.Binary
    logs: vh3.Binary
    json_output: vh3.JSON


@vh3.decorator.external_operation(
    "https://nirvana.yandex-team.ru/operation/8ab677a2-7ba9-4f7c-9492-bbf8fcba7b2e"
)
@vh3.decorator.nirvana_names(
    base_layer="base_layer",
    environment="Environment",
    install_pydl_package="install_pydl_package",
    run_command="run_command",
    openmpi_runner="openmpi_runner",
    openmpi_params="openmpi_params",
    mpi_processes_count="mpi_processes_count",
    before_run_command="before_run_command",
    after_run_command="after_run_command",
    ssh_key="ssh_key",
    auto_snapshot="auto_snapshot",
    nodes_count="nodes_count",
    additional_layers="additional_layers",
    user_requested_secret="user_requested_secret",
)
@vh3.decorator.nirvana_names_transformer(
    vh3.name_transformers.snake_to_dash, options=True
)
def python_3_deep_learning(
    *,
    base_layer: vh3.Enum[
        Literal[
            "PYDL_V3", "PYDL_V4", "PYDL_V5", "PYDL_V5_GPU", "PYDL_V5_GPU_NVIDIA", "NONE"
        ]
    ] = "PYDL_V5_GPU_NVIDIA",
    environment: vh3.MultipleStrings = (),
    pip: vh3.MultipleStrings = (),
    install_pydl_package: vh3.Boolean = True,
    run_command: vh3.String = "python3 $SOURCE_CODE_PATH/__main__.py",
    cpu_cores_usage: vh3.Integer = 1600,
    gpu_count: vh3.Integer = 0,
    openmpi_runner: vh3.Boolean = False,
    openmpi_params: vh3.String = None,
    mpi_processes_count: vh3.Integer = -1,
    before_run_command: vh3.String = None,
    after_run_command: vh3.String = None,
    ttl: vh3.Integer = 1440,
    max_ram: vh3.Integer = 10000,
    max_disk: vh3.Integer = 10000,
    force_tmpfs_disk: vh3.Boolean = False,
    gpu_max_ram: vh3.Integer = 1024,
    gpu_type: vh3.Enum[
        Literal[
            "NONE",
            "ANY",
            "CUDA_ANY",
            "CUDA_3_5",
            "CUDA_5_2",
            "CUDA_6_1",
            "CUDA_7_0",
            "CUDA_8_0",
        ]
    ] = "NONE",
    strict_gpu_type: vh3.Boolean = False,
    ssh_key: vh3.Secret = None,
    yt_token: vh3.Secret = None,
    mr_account: vh3.String = None,
    auto_snapshot: vh3.Integer = 0,
    nodes_count: vh3.Integer = 1,
    additional_layers: vh3.MultipleStrings = (),
    retries_on_job_failure: vh3.Integer = 0,
    timestamp: vh3.Date = "2017-08-25T16:40:26+0300",
    debug_timeout: vh3.Integer = 0,
    user_requested_secret: vh3.Secret = None,
    job_host_tags: vh3.MultipleStrings = (),
    job_scheduler_instance: vh3.String = None,
    job_scheduler_yt_pool_tree: vh3.String = None,
    job_scheduler_yt_pool: vh3.String = None,
    job_scheduler_yt_token: vh3.Secret = None,
    job_mtn_enabled: vh3.Boolean = True,
    job_scheduler_yt_custom_spec: vh3.String = "{}",
    job_layer_yt_path: vh3.MultipleStrings = (),
    job_variables: vh3.MultipleStrings = (),
    script: vh3.Binary = None,
    data: Sequence[vh3.Binary] = (),
    state: vh3.Binary = None,
    volume: Sequence[vh3.Binary] = (),
    params: vh3.JSON = None
) -> Python3DeepLearningOutput:
    """
    Python 3 Deep Learning
    Python code runner with support of tensorflow, numpy, theano, torch, keras and nirvana_dl library. See https://wiki.yandex-team.ru/computervision/projects/deeplearning/nirvanadl/ for user manual.
    :param base_layer: Base porto layer
    :param pip: Libraries to Install
      [[List of libraries to be installed using pip install. Some libraries are already available, see https://wiki.yandex-team.ru/computervision/projects/deeplearning/nirvanadl/ for details. WARNING: use only with explicit version for reproducibility.]]
    :param install_pydl_package: Install nirvana-dl package
    :param run_command: Run command
      [[Custom bash code]]
    :param openmpi_runner: use openmpi runner
    :param openmpi_params: openmpi runner extra args
    :param before_run_command: Before run command
      [[Command which will be executed before run_command]]
    :param after_run_command: After run command
      [[Command which will be executed after run command]]
    :param strict_gpu_type: strict GPU type
    :param ssh_key: SSH Key
      [[Secret with ssh private key to sync logs with remote server]]
    :param yt_token: YT Token
      [[ID of Nirvana Secret with YT access token (https://nda.ya.ru/3RSzVU). Guide to Nirvana Secrets: https://nda.ya.ru/3RSzWZ]]
    :param mr_account: MR Account:
      [[MR Account Name. By default, output tables and directories will be created in some subdirectory of home/<MR Account>/<workflow owner>/nirvana]]
    :param auto_snapshot: Auto Snapshot
      [[Time interval (minutes) to dump snapshots automatically, without explicit python method call. May cause race condition. If 0, option will be disabled]]
    :param nodes_count: Full number of nodes:
      [[Number of nodes. Should be >= 1]]
      Number of nodes. Should be >= 1
    :param additional_layers: Additional porto layers
      [[IDs of porto layers to put on top of base layer]]
    :param timestamp: Timestamp
    :param job_host_tags: master-job-host-tags
    :param job_mtn_enabled:
      https://st.yandex-team.ru/NIRVANA-12358#5eda1eeb6213d14744d9d791
    :param script:
      Tar-archive with source code inside. Unpacked archive will be available by SOURCE_CODE_PATH environment variable.
    :param data:
      Tar archive(s) with various data. Unpacked data will be available by INPUT_PATH environment variable.
      Multiple archives will be concatenated.
    :param state:
      Saved state of the process, will be used as "state" output snapshot.
    :param volume:
      Additional porto volume(s).
    :param params:
      JSON with additional parameters that can be used in your program.
    """
    raise NotImplementedError("Write your local execution stub here")


class GitCheckoutRepoOutput(NamedTuple):
    last_commit: vh3.JSON
    commits_count: vh3.JSON
    archive: vh3.Binary


@vh3.decorator.external_operation(
    "https://nirvana.yandex-team.ru/operation/009eff70-8b27-45b1-b2eb-98fb7b8e5ee7"
)
def git_checkout_repo(
    *,
    ssh_key: vh3.Secret,
    commit_branch: vh3.String = "origin/master",
    dirs: vh3.MultipleStrings = (),
    repo_url: Union[vh3.String, vh3.Text] = None,
    remove_git_files: vh3.Boolean = False,
    only_meta: vh3.Boolean = False,
    recursive_checkout: vh3.Boolean = True,
    top_level_folder: vh3.String = None
) -> GitCheckoutRepoOutput:
    """
    Git: Checkout repo
    Выполняет checkout указанного git/bitbucket репозитория
    :param ssh_key:
      [[private key]]
      private key
    :param commit_branch:
      [[Коммит/ветка/тэг. пример указания ветки - origin/branch_name, пример указания тэга - tags/tag_name]]
      Коммит/ветка/тэг. пример указания ветки - origin/branch_name, пример указания тэга - tags/tag_name
    :param dirs:
      [[Список директорий из репозитория, которые должны попасть в выходной архив, если null - в выходной архив попадет все содержимое репозитория]]
      Список директорий из репозитория, которые должны попасть в выходной архив, если null - в выходной архив попадет все содержимое репозитория
    :param repo_url:
      [[Путь до репозитория в формате ssh://git@server/path/to/repo/repo_name.git]]
      Путь до репозитория в формате ssh://git@server/path/to/repo/repo_name.git
    :param repo_url:
      Путь до репозитория в формате ssh://git@server/path/to/repo/repo_name.git
    :param remove_git_files:
      [[Нужно ли удалить .git директории из выкаченного репозитория]]
      Нужно ли удалить .git директории из выкаченного репозитория
    :param only_meta:
      [[Извлечь только количество коммитов и hash крайнего коммите на ветке без фактического выкачивания репозитория]]
      Извлечь только количество коммитов и hash крайнего коммите на ветке без фактического выкачивания репозитория
    :param recursive_checkout:
      [[Выполнять ли рекурсивный checkout]]
      Выполнять ли рекурсивный checkout
    :param top_level_folder:
      [[Имя top-level директории в архиве(если необходимо специфичное имя), в случае если null, будет выбрано некоторое случайное]]
      Имя top-level директории в архиве(если необходимо специфичное имя), в случае если null, будет выбрано некоторое случайное
    """
    raise NotImplementedError("Write your local execution stub here")


@vh3.decorator.external_operation(
    "https://nirvana.yandex-team.ru/operation/e1e9d91f-7357-4577-b892-eac64c94a3b0"
)
@vh3.decorator.nirvana_output_names("json_output")
@vh3.decorator.nirvana_names_transformer(
    vh3.name_transformers.snake_to_dash, options=True
)
def get_workflow_and_instance_id(
    *, ttl: vh3.Integer = 360, max_ram: vh3.Integer = 100, timestamp: vh3.String = None
) -> vh3.JSON:
    """
    Get workflow and instance id
    """
    raise NotImplementedError("Write your local execution stub here")


@vh3.decorator.external_operation(
    "https://nirvana.yandex-team.ru/operation/961eda13-7e75-4e8e-96aa-fa43c1dd97c1"
)
@vh3.decorator.nirvana_names(max_disk="max-disk")
@vh3.decorator.nirvana_output_names("archive")
def create_tar_archive_10(
    *,
    ttl: vh3.Integer = 360,
    max_disk: vh3.Integer = 1024,
    name0: vh3.String = None,
    name1: vh3.String = None,
    name2: vh3.String = None,
    name3: vh3.String = None,
    name4: vh3.String = None,
    name5: vh3.String = None,
    name6: vh3.String = None,
    name7: vh3.String = None,
    name8: vh3.String = None,
    name9: vh3.String = None,
    tar_mode: vh3.Enum[Literal["none", "extract", "subdir"]] = "extract",
    compress: vh3.Enum[
        Literal[
            "--no-auto-compress",
            "--gzip",
            "--bzip2",
            "--xz",
            "--lzip",
            "--lzma",
            "--lzop",
        ]
    ] = "--no-auto-compress",
    file0: Union[
        vh3.Binary,
        vh3.Executable,
        vh3.HTML,
        vh3.Image,
        vh3.JSON,
        vh3.TSV,
        vh3.Text,
        vh3.XML,
    ] = None,
    file1: Union[
        vh3.Binary,
        vh3.Executable,
        vh3.HTML,
        vh3.Image,
        vh3.JSON,
        vh3.TSV,
        vh3.Text,
        vh3.XML,
    ] = None,
    file2: Union[
        vh3.Binary,
        vh3.Executable,
        vh3.HTML,
        vh3.Image,
        vh3.JSON,
        vh3.TSV,
        vh3.Text,
        vh3.XML,
    ] = None,
    file3: Union[
        vh3.Binary,
        vh3.Executable,
        vh3.HTML,
        vh3.Image,
        vh3.JSON,
        vh3.TSV,
        vh3.Text,
        vh3.XML,
    ] = None,
    file4: Union[
        vh3.Binary,
        vh3.Executable,
        vh3.HTML,
        vh3.Image,
        vh3.JSON,
        vh3.TSV,
        vh3.Text,
        vh3.XML,
    ] = None,
    file5: Union[
        vh3.Binary,
        vh3.Executable,
        vh3.HTML,
        vh3.Image,
        vh3.JSON,
        vh3.TSV,
        vh3.Text,
        vh3.XML,
    ] = None,
    file6: Union[
        vh3.Binary,
        vh3.Executable,
        vh3.HTML,
        vh3.Image,
        vh3.JSON,
        vh3.TSV,
        vh3.Text,
        vh3.XML,
    ] = None,
    file7: Union[
        vh3.Binary,
        vh3.Executable,
        vh3.HTML,
        vh3.Image,
        vh3.JSON,
        vh3.TSV,
        vh3.Text,
        vh3.XML,
    ] = None,
    file8: Union[
        vh3.Binary,
        vh3.Executable,
        vh3.HTML,
        vh3.Image,
        vh3.JSON,
        vh3.TSV,
        vh3.Text,
        vh3.XML,
    ] = None,
    file9: Union[
        vh3.Binary,
        vh3.Executable,
        vh3.HTML,
        vh3.Image,
        vh3.JSON,
        vh3.TSV,
        vh3.Text,
        vh3.XML,
    ] = None
) -> vh3.Binary:
    """
    Create TAR archive 10
    Create TAR archive from given binary files. If the files are archives themselves, unpack them.
    :param name0: File name 0
      [[Ignored if corresponding file is tar archive]]
      Ignored if corresponding file is tar archive
    :param name1: File name 1
      [[Ignored if corresponding file is tar archive]]
      Ignored if corresponding file is tar archive
    :param tar_mode: Tar processing mode
      [[Action for input tar archives]]
      Action for input tar archives
    :param compress: Tar compress
      [[Tar compress mode]]
      Tar compress mode
    :param file0:
      File or archive
    :param file1:
      File or archive
    """
    raise NotImplementedError("Write your local execution stub here")


@vh3.decorator.external_operation(
    "https://nirvana.yandex-team.ru/operation/27620e6e-4e79-487b-bf22-eaeea4ff850f"
)
@vh3.decorator.nirvana_output_names("params")
@vh3.decorator.nirvana_names_transformer(
    vh3.name_transformers.snake_to_dash, options=True
)
def nirvana_api_global_parameters_to_json(
    *,
    token: vh3.Secret,
    workflow: vh3.String,
    max_disk: vh3.Integer = 16,
    timestamp: vh3.Date = None,
    instance: vh3.String = None,
    skip_empty_parameters: vh3.Boolean = False
) -> vh3.JSON:
    """
    Nirvana API global parameters to json
    Get url of input data
    :param token: OAuth token
      [[Nirvana API autorisation token]]
    :param timestamp: Timestamp
    """
    raise NotImplementedError("Write your local execution stub here")


class EchoToTsvJsonTextOutput(NamedTuple):
    output: vh3.TSV
    output_json: vh3.JSON
    output_text: vh3.Text


@vh3.decorator.external_operation(
    "https://nirvana.yandex-team.ru/operation/47a39182-8710-45d6-ae0a-791f741f50e3"
)
@vh3.decorator.nirvana_names_transformer(
    vh3.name_transformers.snake_to_dash, options=True
)
def echo_to_tsv_json_text(
    *, input: vh3.String, ttl: vh3.Integer = 360, job_metric_tag: vh3.String = None
) -> EchoToTsvJsonTextOutput:
    """
    Echo to TSV, JSON, TEXT
    :param input:
      String to copy to output
    :param job_metric_tag:
      Tag for monitoring of resource usage
    """
    raise NotImplementedError("Write your local execution stub here")