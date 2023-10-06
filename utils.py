from __future__ import annotations
from dataclasses import dataclass, astuple
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence, TypeVar
from typing import Generic, NamedTuple
import numpy as np
import numpy.typing as npt

ObsType = TypeVar("ObsType")
AbsObsType = TypeVar("AbsObsType")
ActType = TypeVar("ActType")
AbsActType = TypeVar("AbsActType")


@dataclass(frozen=True, slots=True)
class Transition(tuple, Generic[ObsType, ActType]):
    start_state: ObsType
    action: ActType
    next_state: ObsType
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    def __iter__(self) -> Iterable[Any]:
        return iter(astuple(self))



class Environment(Protocol[ObsType, ActType]):
    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        ...

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        ...


# per fare una tupla di tensori, basta ereditare da tuple
# tuple è immutabile, quindi bisogna sovrascrivere __new__ non __init__
# in breve, quando crei un oggetto, prima chiami __new__ e poi __init__
# __new__ lo chiami dalla classe, crea e ritorna l'oggetto (non inizializzato)
# __init__ lo chiami dall'oggetto, e cambia gli attributi per inizializzarlo
# le cose immutabili, di solito fanno tutto in __new__ (ti ritornano l'oggetto già inizializzato)
# perché non possono essere modificate dopo che l'oggetto è stato creato
# quindi, per fare una tupla di array, basta ereditare da tuple e sovrascrivere __new__ in modo che
# prima chiamare __new__ della superclasse (tuple) converta tutti gli elementi a np.array
# punto bunus, si può mettere le type hints per dire che la tupla contiene array


class MultiArray(tuple[np.ndarray]):
    """A tuple of heterogeneous tensors."""

    def __new__(cls, data: Iterable[npt.ArrayLike]) -> MultiArray:
        return super().__new__(cls, (np.array(el) for el in data))


# se vogliamo associare a ogni array un nome, si potrebbe ereditare da dict
# qui è più facile perché non dobbiamo sovrascrivere __new__
# ma è un po' più complicato per motivi completamente diversi
# idealmente, vorremmo che funzionasse la sintassi: NamedMultiArray(key1=array1, key2=array2, ...)
# ma sta cosa non permette di usare come chiavi le keyword riservate di python
# tipo se volessi avere una chiave chiamata "class" o "import" non posso
# quindi bisognerebbe poter usare anche la sintassi: NamedMultiArray({"key1": array1, "key2": array2, ...})
# per farlo un workaround è usare args e kwargs, ma non è molto elegante
# però è l'unico modo che mi viene in mente per farlo funzionare in modo semplice
# per comodità ho anche aggiunto la possibilita di crearlo da una lista di array e di nomi

# questa è la classe che deve essere costruita con la sintassi: NamedMultiArray(key1=array1, key2=array2, ...)
'''
class NamedMultiArray(dict[str, np.ndarray]):
    """A dictionary of heterogeneous tensors."""

    def __init__(self, *args, **kwargs):
        super().__init__({key: np.array(value) for key, value in kwargs.items()})
'''


# questa è uguale ma aggiunge la possibilità di costruirla con le sintassi:
# NamedMultiArray({"key1": array1, "key2": array2, ...})
# NamedMultiArray(["key1", "key2", ...], [array1, array2, ...],)


class NamedMultiArray(dict[str, np.ndarray]):
    """A dictionary of heterogeneous tensors."""

    def __init__(self, *args, **kwargs):
        """positional argument can be a single dict, or an iterable of names and an iterable of arrays"""
        if len(args) == 0:
            all_arguments = kwargs
        elif len(args) == 1:
            all_arguments = args[0] | kwargs
        elif len(args) == 2:
            all_arguments = dict(zip(args[0], args[1])) | kwargs
        else:
            raise ValueError("Positional arguments can be at most 2")
        super().__init__({key: np.array(value) for key, value in all_arguments.items()})


# è comodo avere un metodo che quando fai partire un option, continua ad iterare la policy
# finché non raggiunge la stop condition. Questo metodo è un generatore, quindi si può usare
# con la sintassi: for transition in iterate_policy(...) e ti cicla tutti risultati

"""def iterate_policy(
    start_state: NamedMultiArray,
    policy: Policy,
    environment: AbstractEnvironment,
) -> Iterable[Transition]:
    action = policy(start_state)
    step = environment.step(action)
    yield Transition(start_state, action, *step)
    
    while not (step.terminated or step.truncated):
        start_state = step.next_state
        action = policy(start_state)
        step = environment.step(action)        
        yield Transition(start_state, action, *step)"""
