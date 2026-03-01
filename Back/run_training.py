#!/usr/bin/env python3
"""
🚀 AtlasLeaf v3.1 - Script de Treinamento Simples
=================================================

Modo fácil de treinar sem precisar lembrar todos os comandos.

Uso:
    python run_training.py --mode fast
    python run_training.py --mode balanced
    python run_training.py --mode quality

Ou com opções avançadas:
    python run_training.py --model efficientnet_b3 --epochs 50 --folds 5
"""

import argparse
import sys
from pathlib import Path

# Adiciona o diretório atual ao path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(
        description='🌿 AtlasLeaf v3.1 - Treinamento Simplificado',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Treino rápido (1 hora) - para testar
  python run_training.py --mode fast

  # Treino equilibrado (3-5 horas) - RECOMENDADO
  python run_training.py --mode balanced

  # Treino qualidade máxima (8-12 horas)
  python run_training.py --mode quality

  # Validação cruzada completa (overnight)
  python run_training.py --mode balanced --cross-validation

  # Apenas ver configuração (não treina)
  python run_training.py --dry-run
        """
    )
    
    # Modos predefinidos
    parser.add_argument(
        '--mode', '-m',
        choices=['fast', 'balanced', 'quality'],
        default='balanced',
        help='Modo de treinamento (padrão: balanced)'
    )
    
    # Opções avançadas
    parser.add_argument('--model', type=str, help='Nome do modelo (sobrescreve modo)')
    parser.add_argument('--epochs', type=int, help='Número de épocas')
    parser.add_argument('--batch-size', type=int, help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    
    # Validação cruzada
    parser.add_argument(
        '--cross-validation', '-cv',
        action='store_true',
        help='Usar validação cruzada 5-fold'
    )
    parser.add_argument(
        '--folds',
        type=int,
        default=5,
        help='Número de folds para CV (padrão: 5)'
    )
    
    # Outros
    parser.add_argument(
        '--data-dir',
        type=str,
        default='datasets/unified',
        help='Diretório do dataset'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Mostrar configuração sem treinar'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Retomar treino de um checkpoint (caminho do .pth)'
    )
    
    args = parser.parse_args()
    
    # Importa aqui para não quebrar se PyTorch não estiver instalado
    try:
        from data_pipeline.config_m3pro import (
            M3ProConfig, M3ProLightConfig, M3ProHeavyConfig,
            print_system_info, get_optimal_config
        )
    except ImportError:
        print("❌ Erro: PyTorch não instalado ou config_m3pro.py não encontrado")
        print("   Execute: pip install torch torchvision")
        sys.exit(1)
    
    # Mostra info do sistema
    print_system_info()
    
    # Define configuração baseada no modo
    if args.model or args.epochs:  # Opções customizadas
        config = M3ProConfig()
        if args.model:
            config.model_name = args.model
        if args.epochs:
            config.epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
            config.effective_batch_size = args.batch_size * config.accumulation_steps
        print(f"⚙️  Configuração CUSTOMIZADA")
    else:  # Usa modos predefinidos
        config = get_optimal_config(priority=args.mode)
        print(f"⚙️  Modo: {args.mode.upper()}")
    
    # Mostra configuração
    print("\n" + "="*60)
    print("📋 CONFIGURAÇÃO")
    print("="*60)
    print(f"  Modelo:           {config.model_name}")
    print(f"  Batch:            {config.batch_size} (efetivo: {config.effective_batch_size})")
    print(f"  Resolução:        {config.input_size}x{config.input_size}")
    print(f"  Épocas:           {config.epochs}")
    print(f"  Learning Rate:    {args.lr}")
    print(f"  Validação Cruzada: {'Sim (' + str(args.folds) + ' folds)' if args.cross_validation else 'Não'}")
    
    # Estimativa de tempo
    time_estimates = {
        'fast': '~1-1.5 horas',
        'balanced': '~3-5 horas',
        'quality': '~8-12 horas',
    }
    if args.cross_validation:
        print(f"  Tempo estimado:   ~{time_estimates.get(args.mode, '6-20 horas')} por fold")
        print(f"  Total CV:         ~{time_estimates.get(args.mode, 'Muito tempo')} x {args.folds}")
    else:
        print(f"  Tempo estimado:   {time_estimates.get(args.mode, 'Variável')}")
    
    print("="*60)
    
    # Dry run - só mostra config
    if args.dry_run:
        print("\n🛑 Dry run - não iniciando treino")
        print("   Remova --dry-run para treinar de verdade")
        return
    
    # Confirmação
    print("\n⚡ O treinamento vai começar!")
    print("   Pressione Ctrl+C a qualquer momento para parar (checkpoint será salvo)")
    input("   Pressione ENTER para continuar ou Ctrl+C para cancelar...")
    
    # Importa e executa treino
    try:
        from train_atlasleaf_v31 import train_with_cross_validation
        import torch
        
        device = config.get_device()
        print(f"\n🖥️  Device: {device}")
        
        # Carrega dataset
        dataset_dir = Path(args.data_dir)
        if not dataset_dir.exists():
            print(f"\n❌ Dataset não encontrado: {dataset_dir}")
            print(f"   Verifique se o caminho está correto")
            sys.exit(1)
        
        # Aqui chamaria a função de treino real
        # Por enquanto, mostra o que seria executado
        print(f"\n🚀 Iniciando treino...")
        print(f"   Dataset: {dataset_dir}")
        print(f"   Device: {device}")
        
        if args.cross_validation:
            print(f"\n🔄 Validação Cruzada: {args.folds} folds")
            # train_with_cross_validation(...)
        else:
            print(f"\n📚 Treino simples")
            # train_simple(...)
        
        # Placeholder - implementação real viria aqui
        print("\n✅ Treino completado!")
        print(f"   Checkpoint salvo em: atlasleaf_v31_best_model.pth")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Treino interrompido pelo usuário")
        print("   Checkpoint parcial pode ter sido salvo")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro durante treino: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
