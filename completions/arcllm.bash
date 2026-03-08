_arcllm_completions() {
  local cur prev words cword
  _init_completion || return

  local commands="start stop restart status logs models health chat env config set set-model completions serve"
  local config_keys="MODEL_ID TORCH_DTYPE API_HOST ROUTER_PORT GPU_IDS WORKER_PORTS DEFAULT_MAX_NEW_TOKENS ENABLE_THINKING MAX_CONCURRENT_REQUESTS MAX_QUEUE_DEPTH"
  local models="Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3.5-27B Qwen/Qwen3.5-35B-A3B"

  if [[ $cword -eq 1 ]]; then
    COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
    return
  fi

  case "${words[1]}" in
    set-model)
      if [[ $prev == "set-model" ]]; then
        COMPREPLY=( $(compgen -W "$models" -- "$cur") )
        return
      fi
      COMPREPLY=( $(compgen -W "--no-restart" -- "$cur") )
      ;;
    set)
      if [[ $prev == "set" ]]; then
        COMPREPLY=( $(compgen -W "$config_keys" -- "$cur") )
        return
      fi
      case "${words[2]}" in
        MODEL_ID) COMPREPLY=( $(compgen -W "$models" -- "$cur") ) ;;
        TORCH_DTYPE) COMPREPLY=( $(compgen -W "float16 bfloat16 float32" -- "$cur") ) ;;
        ENABLE_THINKING) COMPREPLY=( $(compgen -W "true false" -- "$cur") ) ;;
        *) COMPREPLY=( $(compgen -W "--no-restart" -- "$cur") ) ;;
      esac
      ;;
    chat)
      case "$prev" in
        --model) COMPREPLY=( $(compgen -W "$models" -- "$cur") ) ;;
        *) COMPREPLY=( $(compgen -W "--model --max-tokens --temperature" -- "$cur") ) ;;
      esac
      ;;
    logs)
      COMPREPLY=( $(compgen -W "-n --lines -f --follow" -- "$cur") )
      ;;
    completions)
      COMPREPLY=( $(compgen -W "bash zsh" -- "$cur") )
      ;;
  esac
}

complete -F _arcllm_completions arcllm
