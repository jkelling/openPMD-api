"""
This file is part of the openPMD-api.

This module provides functions that are wrapped into sys.exit(...()) calls by
the setuptools (setup.py) "entry_points" -> "console_scripts" generator.

Copyright 2021 openPMD contributors
Authors: Franz Poeschel
License: LGPLv3+
"""
import argparse
import math
import os  # os.path.basename
import re
import sys  # sys.stderr.write
import time

from .. import openpmd_api_cxx as io


class DumpTimes:
    def __init__(self, filename):
        self.last_time_point = int(time.time() * 1000)
        self.out_stream = open(filename, 'w')

    def close(self):
        self.out_stream.close()

    def now(self, description, separator='\t'):
        current = int(time.time() * 1000)
        self.out_stream.write(
            str(current) + separator + str(current - self.last_time_point) +
            separator + description + '\n')
        self.last_time_point = current

    def flush(self):
        self.out_stream.flush()


def parse_args(program_name):
    parser = argparse.ArgumentParser(
        # we need this for line breaks
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
openPMD Pipe.

This tool connects an openPMD-based data source with an openPMD-based data sink
and forwards all data from source to sink.
Possible uses include conversion of data from one backend to another one
or multiplexing the data path in streaming setups.
Parallelization with MPI is optionally possible and is done automatically
as soon as the mpi4py package is found and this tool is called in an MPI
context.
Parallelization with MPI is optionally possible and can be switched on with
the --mpi switch, resp. switched off with the --no-mpi switch.
By default, openpmd-pipe will use MPI if all of the following conditions
are fulfilled:
1) The mpi4py package can be imported.
2) The openPMD-api has been built with support for MPI.
3) The MPI size is greater than 1.
   By default, the openPMD-api will be initialized without an MPI communicator
   if the MPI size is 1. This is to simplify the use of the JSON backend
   which is only available in serial openPMD.
With parallelization enabled, each dataset will be equally sliced according to
a chunk distribution strategy which may be selected via the environment
variable OPENPMD_CHUNK_DISTRIBUTION. Options include "roundrobin",
"binpacking", "slicedataset" and "hostname_<1>_<2>", where <1> should be
replaced with a strategy to be applied within a compute node and <2> with a
secondary strategy in case the hostname strategy does not distribute
all chunks.
The default is `hostname_binpacking_slicedataset`.

Examples:
    {0} --infile simData.h5 --outfile simData_%T.bp
    {0} --infile simData.sst --inconfig @streamConfig.json \\
        --outfile simData_%T.bp
    {0} --infile uncompressed.bp \\
        --outfile compressed.bp --outconfig @compressionConfig.json
""".format(os.path.basename(program_name)))

    parser.add_argument('--infile', type=str, help='In file')
    parser.add_argument('--outfile', type=str, help='Out file')
    parser.add_argument('--inconfig',
                        type=str,
                        default='{}',
                        help='JSON config for the in file')
    parser.add_argument('--outconfig',
                        type=str,
                        default='{}',
                        help='JSON config for the out file')
    # MPI, default: Import mpi4py if available and openPMD is parallel,
    # but don't use if MPI size is 1 (this makes it easier to interact with
    # JSON, since that backend is unavailable in parallel)
    if io.variants['mpi']:
        parser.add_argument('--mpi', action='store_true')
        parser.add_argument('--no-mpi', dest='mpi', action='store_false')
        parser.set_defaults(mpi=None)

    return parser.parse_args()


args = parse_args(sys.argv[0])
# MPI is an optional dependency
if io.variants['mpi'] and (args.mpi is None or args.mpi):
    try:
        from mpi4py import MPI
        HAVE_MPI = True
    except (ImportError, ModuleNotFoundError):
        if args.mpi:
            raise
        else:
            print("""
    openPMD-api was built with support for MPI,
    but mpi4py Python package was not found.
    Will continue in serial mode.""",
                  file=sys.stderr)
            HAVE_MPI = False
else:
    HAVE_MPI = False

debug = False


class FallbackMPICommunicator:
    def __init__(self):
        self.size = 1
        self.rank = 0


class deferred_load:
    def __init__(self, source, dynamicView, offset, extent):
        self.source = source
        self.dynamicView = dynamicView
        self.offset = offset
        self.extent = extent


# Example how to implement a simple partial strategy in Python
class LoadOne(io.PartialStrategy):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def assign(self, assignment, *_):
        element = assignment.not_assigned.pop()
        if self.rank not in assignment.assigned:
            assignment.assigned[self.rank] = [element]
        else:
            assignment.assigned[self.rank].append(element)
        return assignment

class IncreaseGranularity(io.PartialStrategy):
    def __init__(
        self,
        granularity_in,
        granularity_out,
        inner_distribution=io.ByHostname(io.RoundRobin()),
    ):
        super().__init__()
        self.inner_distribution = inner_distribution
        self.granularity_in = granularity_in
        self.granularity_out = granularity_out

    def assign(self, assignment, in_ranks, out_ranks):
        if "in_ranks_inner" in dir(self):
            return self.inner_distribution.assign(
                assignment, self.in_ranks_inner, self.out_ranks_inner
            )

        def hosts_in_order(rank_assignment):
            already_seen = set()
            res = []
            for (_, hostname) in rank_assignment.items():
                if hostname not in already_seen:
                    already_seen.add(hostname)
                    res.append(hostname)
            return res

        in_hosts_in_order = hosts_in_order(in_ranks)
        out_hosts_in_order = hosts_in_order(out_ranks)

        def hostname_to_hostgroup(ordered_hosts, granularity):
            res = {}  # real host -> host group
            current_meta_host = 0
            granularity_counter = 0
            for host in ordered_hosts:
                res[host] = str(current_meta_host)
                granularity_counter += 1
                if granularity_counter >= granularity:
                    granularity_counter = 0
                    current_meta_host += 1
            return res

        in_hostname_to_hostgroup = hostname_to_hostgroup(
            in_hosts_in_order, self.granularity_in
        )
        out_hostname_to_hostgroup = hostname_to_hostgroup(
            out_hosts_in_order, self.granularity_out
        )

        def inner_rank_assignment(outer_assignment, hostname_to_hostgroup):
            res = {}
            for (rank, hostname) in outer_assignment.items():
                res[rank] = hostname_to_hostgroup[hostname]
            return res

        self.in_ranks_inner = inner_rank_assignment(in_ranks, in_hostname_to_hostgroup)
        self.out_ranks_inner = inner_rank_assignment(
            out_ranks, out_hostname_to_hostgroup
        )

        return self.inner_distribution.assign(
            assignment, self.in_ranks_inner, self.out_ranks_inner
        )

class MergingStrategy(io.Strategy):
    def __init__(self, inner_strategy):
        super().__init__()
        self.inner_strategy = inner_strategy

    def assign(self, assignment, in_ranks, out_ranks):
        res = self.inner_strategy.assign(assignment, in_ranks, out_ranks)
        for out_rank, assignment in res.items():
            merged = assignment.merge_chunks_from_same_sourceID()
            assignment.clear()
            for in_rank, chunks in merged.items():
                for chunk in chunks:
                    assignment.append(
                        io.WrittenChunkInfo(chunk.offset, chunk.extent, in_rank)
                    )
        return res


# strategy = IncreaseGranularity(2, 1)
# assignment = [
#     io.WrittenChunkInfo([0], [1], 0),
#     io.WrittenChunkInfo([1], [1], 1),
#     io.WrittenChunkInfo([2], [1], 2),
#     io.WrittenChunkInfo([3], [1], 3),
# ]
# in_ranks = {0: "host0", 1: "host1", 2: "host3", 3: "host4"}
# out_ranks = {0: "host2", 1: "host5"}
# res = strategy.assign(assignment, in_ranks, out_ranks)
# print(f"NOT ASSIGNED: {len(res.not_assigned)} chunks")
# print("ASSIGNED:")
# for rank, chunks in res.assigned.items():
#     print(f"\tRANK {rank}:", end='')
#     for chunk in chunks:
#         print(f" [{chunk.offset}-{chunk.extent}]", end='')
#     print()

#Example how to implement a simple strategy in Python
class LoadAll(io.Strategy):

    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def assign(self, assignment, *_):
        res = assignment.assigned
        if self.rank not in res:
            res[self.rank] = assignment.not_assigned
        else:
            res[self.rank].extend(assignment.not_assigned)
        return res


def distribution_strategy(dataset_extent,
                          mpi_rank,
                          mpi_size,
                          strategy_identifier=None):
    if strategy_identifier is None or not strategy_identifier:
        if 'OPENPMD_CHUNK_DISTRIBUTION' in os.environ:
            strategy_identifier = os.environ[
                'OPENPMD_CHUNK_DISTRIBUTION'].lower()
        else:
            strategy_identifier = 'hostname_binpacking_slicedataset'  # default
    match = re.search('hostname_(.*)_(.*)', strategy_identifier)
    if match is not None:
        inside_node = distribution_strategy(dataset_extent,
                                            mpi_rank,
                                            mpi_size,
                                            strategy_identifier=match.group(1))
        second_phase = distribution_strategy(
            dataset_extent,
            mpi_rank,
            mpi_size,
            strategy_identifier=match.group(2))
        return io.FromPartialStrategy(io.ByHostname(inside_node), second_phase)
    elif strategy_identifier == 'fan_in':
        granularity = os.environ['OPENPMD_FAN_IN']
        granularity = int(granularity)
        return IncreaseGranularity(
            granularity, 1,
            io.FromPartialStrategy(io.ByHostname(io.RoundRobin()),
                                   io.DiscardingStrategy()))
    elif strategy_identifier == 'all':
        return io.FromPartialStrategy(IncreaseGranularity(5), LoadAll(mpi_rank))
    elif strategy_identifier == 'roundrobin':
        return io.RoundRobin()
    elif strategy_identifier == 'binpacking':
        return io.BinPacking()
    elif strategy_identifier == 'slicedataset':
        return io.ByCuboidSlice(io.OneDimensionalBlockSlicer(), dataset_extent,
                                mpi_rank, mpi_size)
    elif strategy_identifier == 'fail':
        return io.FailingStrategy()
    elif strategy_identifier == 'discard':
        return io.DiscardingStrategy()
    else:
        raise RuntimeError("Unknown distribution strategy: " +
                           strategy_identifier)


class pipe:
    """
    Represents the configuration of one "pipe" pass.
    """
    def __init__(self, infile, outfile, inconfig, outconfig, comm):
        self.infile = infile
        self.outfile = outfile
        self.inconfig = inconfig
        self.outconfig = outconfig
        self.loads = []
        self.comm = comm
        if HAVE_MPI:
            hostinfo = io.HostInfo.MPI_PROCESSOR_NAME
            self.outranks = hostinfo.get_collective(self.comm)
            my_hostname = self.outranks[self.comm.rank]
            self.outranks = {i: rank for i, rank in self.outranks.items() if rank == my_hostname}
        else:
            self.outranks = {i: str(i) for i in range(self.comm.size)}

    def run(self, loggingfile):
        if not HAVE_MPI or (args.mpi is None and self.comm.size == 1):
            print("Opening data source")
            sys.stdout.flush()
            inseries = io.Series(self.infile, io.Access.read_linear,
                                 self.inconfig)
            print("Opening data sink")
            sys.stdout.flush()
            outseries = io.Series(self.outfile, io.Access.create,
                                  self.outconfig)
            print("Opened input and output")
            sys.stdout.flush()
        else:
            print("Opening data source on rank {}.".format(self.comm.rank))
            sys.stdout.flush()
            inseries = io.Series(self.infile, io.Access.read_linear, self.comm,
                                 self.inconfig)
            print("Opening data sink on rank {}.".format(self.comm.rank))
            sys.stdout.flush()
            outseries = io.Series(self.outfile, io.Access.create, self.comm,
                                  self.outconfig)
            print("Opened input and output on rank {}.".format(self.comm.rank))
            sys.stdout.flush()
        dump_times = DumpTimes(loggingfile)
        # In Linear read mode, global attributes are only present after calling
        # this method to access the first iteration
        inseries.parse_base()
        self.__copy(inseries, outseries, dump_times)
        dump_times.close()
        del inseries
        del outseries

    def __copy(self, src, dest, dump_times, current_path="/data/"):
        """
        Worker method.
        Copies data from src to dest. May represent any point in the openPMD
        hierarchy, but src and dest must both represent the same layer.
        """
        if (type(src) is not type(dest)
                and not isinstance(src, io.IndexedIteration)
                and not isinstance(dest, io.Iteration)):
            raise RuntimeError(
                "Internal error: Trying to copy mismatching types")
        attribute_dtypes = src.attribute_dtypes
        # The following attributes are written automatically by openPMD-api
        # and should not be manually overwritten here
        ignored_attributes = {
            io.Series:
            ["basePath", "iterationEncoding", "iterationFormat", "openPMD"],
            io.Iteration: ["snapshot"],
            io.Record_Component: ["value", "shape"] if isinstance(
                src, io.Record_Component) and src.constant else []
        }
        # filter the map for relevant openpmd object model types
        from itertools import chain
        ignored_attributes = set(chain.from_iterable(value for (
            key, value) in ignored_attributes.items() if isinstance(src, key)))

        for key in src.attributes:
            ignore_this_attribute = key in ignored_attributes
            if not ignore_this_attribute:
                attr = src.get_attribute(key)
                attr_type = attribute_dtypes[key]
                dest.set_attribute(key, attr, attr_type)

        container_types = [
            io.Mesh_Container, io.Particle_Container, io.ParticleSpecies,
            io.Record, io.Mesh, io.Particle_Patches, io.Patch_Record
        ]
        is_container = any([
            isinstance(src, container_type)
            for container_type in container_types
        ])

        if isinstance(src, io.Series):
            # main loop: read iterations of src, write to dest
            write_iterations = dest.write_iterations()
            for in_iteration in src.read_iterations():
                dump_times.now("Received iteration {}".format(
                    in_iteration.iteration_index))
                dump_times.flush()
                if self.comm.rank == 0:
                    print("Iteration {0} contains {1} meshes:".format(
                        in_iteration.iteration_index,
                        len(in_iteration.meshes)))
                    for m in in_iteration.meshes:
                        print("\t {0}".format(m))
                    print("")
                    print(
                        "Iteration {0} contains {1} particle species:".format(
                            in_iteration.iteration_index,
                            len(in_iteration.particles)))
                    for ps in in_iteration.particles:
                        print("\t {0}".format(ps))
                        print("With records:")
                        for r in in_iteration.particles[ps]:
                            print("\t {0}".format(r))
                # With linear read mode, we can only load the source rank table
                # inside `read_iterations()` since it's a dataset.
                self.inranks = src.get_rank_table(collective=True)
                out_iteration = write_iterations[in_iteration.iteration_index]
                sys.stdout.flush()
                loadedbytes = self.__copy(
                    in_iteration, out_iteration, dump_times,
                    current_path + str(in_iteration.iteration_index) + "/")
                for deferred in self.loads:
                    deferred.source.load_chunk(
                        deferred.dynamicView.current_buffer(), deferred.offset,
                        deferred.extent)
                dump_times.now(
                    "Closing incoming iteration {} to load {} bytes".format(
                     in_iteration.iteration_index, loadedbytes))
                dump_times.flush()
                in_iteration.close()
                dump_times.now("Closing outgoing iteration {}".format(
                    in_iteration.iteration_index))
                dump_times.flush()
                out_iteration.close()
                dump_times.now("Closed outgoing iteration {}".format(
                    in_iteration.iteration_index))
                dump_times.flush()
                self.loads.clear()
                sys.stdout.flush()
        elif isinstance(src, io.Record_Component) and (not is_container
                                                       or src.scalar):
            shape = src.shape
            dtype = src.dtype
            dest.reset_dataset(io.Dataset(dtype, shape))
            if src.empty:
                # empty record component automatically created by
                # dest.reset_dataset()
                return 0
            elif src.constant:
                dest.make_constant(src.get_attribute("value"))
                return 0
            else:
                chunk_table = src.available_chunks()
                # todo buffer the strategy
                strategy = distribution_strategy(shape, self.comm.rank,
                                                 self.comm.size)
                my_chunks = strategy.assign(chunk_table, self.inranks,
                                            self.outranks)
                accum = 0
                for chunk in my_chunks[
                        self.comm.rank] if self.comm.rank in my_chunks else []:
                    if debug:
                        end = chunk.offset.copy()
                        for i in range(len(end)):
                            end[i] += chunk.extent[i]
                        print("{}\t{}/{}:\t{} -- {}".format(
                            current_path, self.comm.rank, self.comm.size,
                            chunk.offset, end))
                    accum += math.prod(chunk.extent)
                    span = dest.store_chunk(chunk.offset, chunk.extent)
                    self.loads.append(
                        deferred_load(src, span, chunk.offset, chunk.extent))

                accum *= dtype.itemsize
                # print(accum, "Bytes for", current_path)
                return accum

        elif isinstance(src, io.Iteration):
            # m = self.__copy(src.meshes, dest.meshes, dump_times,
            #             current_path + "meshes/")
            p = self.__copy(src.particles, dest.particles, dump_times,
                        current_path + "particles/")
            return p
        elif is_container:
            acc = 0
            for key in src:
                acc += self.__copy(src[key], dest[key], dump_times,
                            current_path + key + "/")
            # if isinstance(src, io.ParticleSpecies):
            #     self.__copy(src.particle_patches, dest.particle_patches,
            #                 dump_times)
            return acc
        else:
            raise RuntimeError("Unknown openPMD class: " + str(src))


def main():
    if not args.infile or not args.outfile:
        print("Please specify parameters --infile and --outfile.")
        sys.exit(1)
    if HAVE_MPI:
        communicator = MPI.COMM_WORLD
    else:
        communicator = FallbackMPICommunicator()
    run_pipe = pipe(args.infile, args.outfile, args.inconfig, args.outconfig,
                    communicator)

    max_logs = 20
    stride = (communicator.size + max_logs) // max_logs - 1  # sdiv, ceil(a/b)
    if stride == 0:
        stride += 1
    if communicator.rank % stride == 0:
        loggingfile = "./PIPE_times_{}.txt".format(communicator.rank)
    else:
        loggingfile = "/dev/null"
    print("Logging file on rank {} of {} is \"{}\".".format(communicator.rank, communicator.size, loggingfile))

    run_pipe.run(loggingfile)


if __name__ == "__main__":
    main()
    sys.exit()
