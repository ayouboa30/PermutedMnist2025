import sys
import os
import json
import time
import numpy as np
import importlib
import tracemalloc
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from permuted_mnist.env.permuted_mnist import PermutedMNISTEnv
except ImportError:
    print("Erreur: Le dossier 'permuted_mnist' est introuvable ou l'environnement n'a pas pu être chargé.", file=sys.stderr)
    sys.exit(1)

AGENT_FOLDER = "agents"
SUBMISSION_FOLDER = "soumissions"
SEED = 42
NUMBER_EPISODES = 7

ALL_AGENT_NAMES = [
    'BestAgentMLP',
    'ExtraTreeAgent5features',
    'ExtraTreeAgent9features',
    'KNNFais',
    'LGBEXtHybride',
    'LGRMAgent5features',
    'MLPAgentBase',
    'MLPBoost10features',
    'MLPV2_1'
]

def import_agent_class(agent_name: str):
    """
    Importe dynamiquement la classe 'Agent' depuis le module 'agents.{agent_name}'.
    """
    try:
        module_path = f"agents.{agent_name}"
        module = importlib.import_module(module_path)
        
        if hasattr(module, "Agent"):
            return getattr(module, "Agent")
        else:
            print(f"Problème avec l'agent '{agent_name}': Le module '{module_path}' a été trouvé, mais il ne contient pas de 'class Agent'.", file=sys.stderr)
            return None
            
    except ImportError as e:
        print(f"Problème avec l'agent '{agent_name}': Impossible d'importer le fichier '{module_path}'. Vérifiez le nom du fichier et s'il contient des erreurs. ({e})", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Problème avec l'agent '{agent_name}': Une erreur s'est produite lors de l'import : {e}", file=sys.stderr)
        return None

def run_single_agent(agent_name: str):
    """
    Exécute l'évaluation complète pour un seul agent et sauvegarde les résultats.
    """
    print("\n" + "="*60)
    print(f"ÉVALUATION DE L'AGENT : {agent_name}")
    print("="*60 + "\n")

    AgentClass = import_agent_class(agent_name)
    if AgentClass is None:
        return

    env = PermutedMNISTEnv(number_episodes=NUMBER_EPISODES)
    env.set_seed(SEED)
    
    try:
        agent = AgentClass(output_dim=10, seed=SEED)
    except Exception as e:
        print(f"Problème avec l'agent '{agent_name}': Impossible de l'initialiser (vérifiez __init__). Erreur: {e}", file=sys.stderr)
        return

    accuracies = []
    times = []
    cpu_times = []
    task_num = 1

    tracemalloc.start()

    while True:
        task = env.get_next_task()
        if task is None:
            break
        
        print(f"--- Tâche {task_num}/{NUMBER_EPISODES} ---")
        
        try:
            start_time = time.time()
            cpu_start = time.process_time()

            agent.train(task['X_train'], task['y_train'])
            predictions = agent.predict(task['X_test'])

            elapsed_time = time.time() - start_time
            cpu_time = time.process_time() - cpu_start

            accuracy = env.evaluate(predictions, task['y_test'])
            
            accuracies.append(accuracy)
            times.append(elapsed_time)
            cpu_times.append(cpu_time)
            
            print(f"Tâche {task_num}: Précision = {accuracy:.2%}, Temps = {elapsed_time:.4f}s, CPU = {cpu_time:.4f}s")
        
        except Exception as e:
            print(f"!!! L'agent {agent_name} a planté pendant la tâche {task_num} !!!", file=sys.stderr)
            print(f"Détail de l'erreur: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            accuracies.append(0.0)
            times.append(elapsed_time if 'elapsed_time' in locals() else 0)
            cpu_times.append(cpu_time if 'cpu_time' in locals() else 0)
        
        task_num += 1

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024 * 1024)

    if not accuracies:
        print(f"Aucune tâche n'a pu être terminée pour {agent_name}.")
        return

    results = {
        "agent_name": agent_name,
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "total_time": np.sum(times),
        "total_cpu_time": np.sum(cpu_times),
        "peak_memory_mb": peak_mem_mb,
        "accuracies_per_task": accuracies,
        "times_per_task": times,
        "cpu_times_per_task": cpu_times
    }
    
    os.makedirs(SUBMISSION_FOLDER, exist_ok=True)
    submission_file = os.path.join(SUBMISSION_FOLDER, f"{agent_name}_results.json")
    
    try:
        with open(submission_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nRésultats sauvegardés dans : {submission_file}")
    except Exception as e:
        print(f"Impossible d'écrire le fichier de résultats pour {agent_name}: {e}", file=sys.stderr)

    print(f"\nRésumé pour {agent_name}:")
    print(f"  Précision moyenne: {results['mean_accuracy']:.2%} ± {results['std_accuracy']:.2%}")
    print(f"  Temps total (réel): {results['total_time']:.2f}s")
    print(f"  Temps total (CPU) : {results['total_cpu_time']:.2f}s")
    print(f"  Pic de RAM          : {results['peak_memory_mb']:.2f} MB")


if __name__ == "__main__":
    
    if len(sys.argv) == 2:
        agent_to_run = sys.argv[1]
        print(f"Mode 1: Exécution d'un seul agent : {agent_to_run}")
        run_single_agent(agent_to_run)
        
    elif len(sys.argv) == 1:
        print(f"Mode 2: Exécution de TOUS les {len(ALL_AGENT_NAMES)} agents...")
        overall_start_time = time.time()
        for agent_name in ALL_AGENT_NAMES:
            run_single_agent(agent_name)
        overall_elapsed_time = time.time() - overall_start_time
        print("\n" + "="*60)
        print(f"BENCHMARK COMPLET TERMINÉ en {overall_elapsed_time:.2f}s")
        
    else:
        print("Usage:", file=sys.stderr)
        print(f"  python {sys.argv[0]}            (Pour exécuter tous les agents)", file=sys.stderr)
        print(f"  python {sys.argv[0]} [AgentName] (Pour exécuter un seul agent)", file=sys.stderr)
        sys.exit(1)