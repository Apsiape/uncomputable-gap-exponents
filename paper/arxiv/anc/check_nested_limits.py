"""Independent nested-limit checks. No supplied code is imported; one process/thread."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import itertools
import json
import math
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
NL, NR, INC, RET = range(4)

def counter(radices, alphabet):
    """Enumerate the complete product basis and retain local-term multiplicities."""
    width = len(radices)
    local = 4 * alphabet
    states = list(itertools.product(range(local), repeat=width))
    index = {state: i for i, state in enumerate(states)}
    penalties = []
    heads = []
    edges = defaultdict(int)
    for i, state in enumerate(states):
        digits = [v // 4 for v in state]
        symbols = [v % 4 for v in state]
        heads.append(sum(v >= INC for v in symbols))
        penalty = sum(a >= b for a, b in zip(digits, radices))
        penalty += sum((a == NR and b == NL) or (a >= INC and b == NL)
                       or (a == NR and b >= INC) or (a >= INC and b >= INC)
                       for a, b in zip(symbols, symbols[1:]))
        penalties.append(penalty)
        def term(changes):
            other = list(state)
            for position, digit, symbol in changes:
                other[position] = 4 * digit + symbol
            j = index[tuple(other)]
            assert i != j
            edges[tuple(sorted((i, j)))] += 1
        for j, (digit, symbol, radix) in enumerate(zip(digits, symbols, radices)):
            if symbol == INC:
                if digit < radix - 1:
                    term([(j, digit + 1, RET)])
                elif digit == radix - 1:
                    if j + 1 == width:
                        term([(j, 0, RET)])
                    elif symbols[j + 1] == NR:
                        term([(j, 0, NL), (j + 1, digits[j + 1], INC)])
            if symbol == RET and j == 0:
                term([(j, digit, INC)])
            if j + 1 < width and symbol == NL and digit == 0 and symbols[j + 1] == RET:
                term([(j, 0, RET), (j + 1, digits[j + 1], NR)])
    adjacency = [[] for _ in states]
    for (a, b), weight in edges.items():
        adjacency[a].append((b, weight))
        adjacency[b].append((a, weight))
    assert all(len(adjacency[i]) <= 2 for i, h in enumerate(heads) if h == 1)
    assert all(penalties[i] >= 1 for i, h in enumerate(heads) if h >= 2)
    seen = set()
    positive = math.inf
    components = 0
    max_block = 0
    for initial in range(len(states)):
        if initial in seen:
            continue
        stack, block = [initial], []
        seen.add(initial)
        while stack:
            vertex = stack.pop()
            block.append(vertex)
            for other, _ in adjacency[vertex]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        components += 1
        if len(block) == 1:
            if penalties[initial]:
                positive = min(positive, penalties[initial])
            continue
        if all(penalties[i] for i in block) and positive <= 1:
            continue  # exact operator bound L + V >= I
        max_block = max(max_block, len(block))
        assert len(block) < 3000
        loc = {v: i for i, v in enumerate(block)}
        matrix = np.diag([float(penalties[v] + sum(w for _, w in adjacency[v])) for v in block])
        for vertex in block:
            for other, weight in adjacency[vertex]:
                matrix[loc[vertex], loc[other]] -= weight
        eigenvalues = np.linalg.eigvalsh(matrix)
        assert eigenvalues[0] >= -1e-10
        excited = eigenvalues[eigenvalues > 1e-8]
        if len(excited):
            positive = min(positive, float(excited[0]))
    product = math.prod(radices)
    length = 2 * sum(product // math.prod(radices[:j]) for j in range(width))
    cycle_gap = 4 * math.sin(math.pi / length) ** 2
    lower = 1 / (64 * width * width * product * product)
    upper = 4 * math.pi * math.pi / (product * product)
    assert lower <= positive + 1e-10 <= upper + 1e-10
    return dict(radices=radices, alphabet=alphabet, dimension=len(states), gap=positive,
                cycle_length=length, cycle_gap=cycle_gap, equals_cycle=abs(positive-cycle_gap) < 1e-8,
                lower=lower, upper=upper, components=components, max_diagonalized_block=max_block,
                double_edge=max(edges.values(), default=0) > 1)

def flags():
    tested = 0
    for sizes in ((1, 1, 1), (1, 2, 3), (2, 2, 2), (3, 2, 2)):
        sites = list(itertools.product(*(range(n) for n in sizes)))
        for axis in range(3):
            for side in (0, 1):
                correct = [int(v[axis] == (sizes[axis]-1 if side else 0)) for v in sites]
                for word in itertools.product((0, 1), repeat=len(sites)):
                    energy = sum(bit * (-1 if face else 1) for bit, face in zip(word, correct))
                    distance = sum(a != b for a, b in zip(word, correct))
                    assert energy == -sum(correct) + distance
                    tested += 1
    return dict(flag_assignments=tested, all_pass=True)

def legal_counter_components(radices):
    """Independently enumerate just the invariant penalty-free one-head subspace."""
    width = len(radices)
    vertices = [(digits, pos, mode) for digits in itertools.product(*(range(r) for r in radices))
                for pos in range(width) for mode in (INC, RET)]
    graph = {v: defaultdict(int) for v in vertices}
    for vertex in vertices:
        digits, pos, mode = vertex
        updated = list(digits)
        other = None
        if mode == INC:
            if digits[pos] < radices[pos]-1:
                updated[pos] += 1
                other = (tuple(updated), pos, RET)
            elif pos == width-1:
                updated[pos] = 0
                other = (tuple(updated), pos, RET)
            else:
                updated[pos] = 0
                other = (tuple(updated), pos+1, INC)
        elif pos == 0:
            other = (digits, pos, INC)
        elif digits[pos-1] == 0:
            other = (digits, pos-1, RET)
        if other is not None:
            graph[vertex][other] += 1
            graph[other][vertex] += 1
    seen, components = set(), []
    for vertex in vertices:
        if vertex in seen:
            continue
        todo, block = [vertex], []
        seen.add(vertex)
        while todo:
            current = todo.pop()
            block.append(current)
            for other in graph[current]:
                if other not in seen:
                    seen.add(other)
                    todo.append(other)
        n = len(block)
        cycle = all(sum(graph[v].values()) == 2 for v in block)
        if cycle:
            gap = 4*math.sin(math.pi/n)**2
        else:
            assert sum(len(graph[v]) == 1 for v in block) == 2
            gap = 4*math.sin(math.pi/(2*n))**2
        components.append(dict(size=n, cycle=cycle, gap=gap))
    return components

def thickness():
    tested, cells = 0, 0
    for size in list(range(1, 130)) + [255, 256, 257, 511, 512, 513, 1023, 1024, 1025]:
        for width in (1, 2, (size-1).bit_length()+2):
            bits, past = [0]*width, [0]*width
            for row in range(size):
                assert sum(b << j for j, b in enumerate(bits)) == row % (1 << width)
                assert [int(a or b) for a, b in zip(bits, past)] == [int(row >= (1 << j)) for j in range(width)]
                carry = [1]
                for j in range(width-1):
                    carry.append(bits[j] & carry[-1])
                past, bits = [a | b for a, b in zip(past, bits)], [a ^ c for a, c in zip(bits, carry)]
                cells += width
            tested += 1
    # Explicit self-delimiting read-only inputs for the higher-dimensional schema.
    reads = 0
    for value in range(257):
        width = value.bit_length()+2
        track = [((value >> j) & 1, int(value >= 1 << j)) for j in range(width)]
        recovered = 0
        for j, (bit, active) in enumerate(track):
            if not active:
                break
            recovered += bit << j
        assert recovered == value
        reads += 1
    return dict(counter_boxes=tested, checked_cells=cells, delimited_inputs=reads, all_pass=True)

def density():
    tested = 0
    for mode in range(5):
        start, ones = 1, 0
        for stage in range(45):
            end = (stage+2)*start
            slope = (Fraction(0), Fraction(1), Fraction(1, 3), Fraction(stage % 2, 1),
                     Fraction(1 if stage % 2 else 3, 4))[mode]
            total = end-start
            for offset in sorted({0, 1, total//3, total//2, max(0,total-1), total}):
                prefix = ones + (offset*slope.numerator)//slope.denominator
                density_now = Fraction(prefix, start+offset)
                interpolated = Fraction(ones, start+offset) + Fraction(offset, start+offset)*slope
                assert abs(density_now-interpolated) <= Fraction(1, start+offset)
                tested += 1
            ones += (total*slope.numerator)//slope.denominator
            assert abs(Fraction(ones,end)-slope) <= Fraction(1,stage+2)+Fraction(1,end)
            start = end
    return dict(exact_fraction_prefix_checks=tested, all_pass=True)

def stabilization():
    # A total, fair sequence of physical write events with repeated mind changes.
    limit_bits = [1 if j < 3 else j % 2 for j in range(81)]
    events = []
    tick = 0
    for stage in range(81):
        for j in range(stage+1):
            tick += 1 + (j+stage) % 7
            value = limit_bits[j] if j < 3 or stage >= 3*j+5 else (limit_bits[j] ^ (stage % 2))
            events.append((tick, j, value))
    last_wrong = [0]*12
    first_correct_after = [None]*12
    for when, j, value in events:
        if j < 12 and value != limit_bits[j]:
            last_wrong[j] = when
    for when, j, value in events:
        if j < 12 and when > last_wrong[j] and value == limit_bits[j] and first_correct_after[j] is None:
            first_correct_after[j] = when
    cases = []
    for size in (5, 6, 8, 9, 16, 17, 32, 33, 64, 65):
        width = (size-1).bit_length()
        threshold = max(width+2, 1+max(first_correct_after[:width]))
        current = [0]*81
        samples = 0
        for when, j, value in events:
            current[j] = value
            if when >= threshold-1:
                assert current[:width] == limit_bits[:width]
                samples += 1
        cases.append(dict(M=size, ell=width, L0=threshold, tested_later_events=samples))
    return dict(cases=cases, all_pass=True)

def calculus_witnesses():
    # Computable real names whose strict zero comparisons encode delayed halting.
    approximations = 0
    for halt_time in list(range(1,31))+[None]:
        true_value = Fraction(0) if halt_time is None else Fraction(1,2**halt_time)
        for stage in range(31):
            approx = true_value if halt_time is not None and halt_time <= stage else Fraction(0)
            assert abs(approx-true_value) <= Fraction(1,2**stage)
            approximations += 1
    # Uniformly effectively convergent convex p_n need not preserve second responses.
    for n in (2, 4, 8, 16, 32):
        for x in (-.8, -.2, 0, .3, 1.):
            p_n = x*x + (1-math.cos(n*x))/(n*n)
            assert abs(p_n-x*x) <= 2/(n*n)+1e-15
            assert 2+math.cos(n*x) >= 1-1e-15
        assert 2+math.cos(0) == 3  # p_n''(0)=3, whereas p_inf''(0)=2
    return dict(computable_name_error_checks=approximations, second_response_counterexample=True, all_pass=True)

def main():
    started = time.time()
    result = {}
    records = []
    for alphabet, allowed, maximum in ((2, (1,2), 4), (4, (1,2,4), 3)):
        for width in range(1, maximum+1):
            for radices in itertools.product(allowed, repeat=width):
                record = counter(list(radices), alphabet)
                records.append(record)
    for radices in ([2]*5, [2,1,2,1,2], [1]*5):
        records.append(counter(radices, 2))
    result['counters'] = records
    print('COUNTERS', len(records), 'PASS; cycle-equality counterexamples', sum(not x['equals_cycle'] for x in records), flush=True)
    for record in records:
        components = legal_counter_components(record['radices'])
        prediction = min(c['gap'] for c in components)
        if len(record['radices']) >= 2 or any(r < record['alphabet'] for r in record['radices']):
            prediction = min(1, prediction)
        assert abs(prediction-record['gap']) < 1e-8
    late_radices = [2,2,2]+[1]*15
    components = legal_counter_components(late_radices)
    cycle_length = 2*sum(math.prod(late_radices[j:]) for j in range(len(late_radices)))
    cycle_gap = 4*math.sin(math.pi/cycle_length)**2
    other_gap = min(c['gap'] for c in components)
    assert cycle_length == 58 and any(c['size'] == 30 and not c['cycle'] for c in components)
    assert other_gap < cycle_gap
    result['allowed_prefix_cycle_counterexample'] = dict(radices=late_radices, cycle_length=cycle_length,
        cycle_gap=cycle_gap, actual_gap=min(1,other_gap), components=components,
        M=131074, explanation='A={0,1,2}, beta=1, ell(M)=18; a closed 30-vertex legal path undercuts the main 58-cycle.')
    print('PREFIX COUNTEREXAMPLE', cycle_gap, other_gap, flush=True)
    for name, check in (('flags',flags),('thickness',thickness),('density',density),
                        ('stabilization',stabilization),('calculus',calculus_witnesses)):
        result[name] = check()
        print(name.upper(), json.dumps(result[name]), flush=True)
    result['seconds'] = time.time()-started
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class MemoryCounters(ctypes.Structure):
            _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
                        ('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),
                        ('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
        memory = MemoryCounters()
        memory.cb = ctypes.sizeof(memory)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD]
        ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                                       ctypes.byref(memory), memory.cb)
        result['peak_working_set_bytes'] = memory.PeakWorkingSetSize
        assert memory.PeakWorkingSetSize < 1_000_000_000
    result['all_pass'] = True
    (HERE/'check_nested_limits.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('ALL PASS', result['seconds'], flush=True)

if __name__ == '__main__':
    main()
