import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class NoC:
    """
    Represents the Network-on-Chip (NoC) architecture and provides methods
    for simulating neural network execution.
    """

    def __init__(self, topology="2D_mesh", dimensions=(4, 4), buffer_size=1024, routing_algo="XY",
                 fault_tolerance_enabled=False, error_detection_method="EDC",
                 error_recovery_method="retransmission",
                 pe_count = None):
        """
        Initializes the NoC.
        """
        self.topology = topology
        self.dimensions = dimensions
        self.buffer_size = buffer_size
        self.routing_algo = routing_algo
        self.fault_tolerance_enabled = fault_tolerance_enabled
        self.error_detection_method = error_detection_method
        self.error_recovery_method = error_recovery_method


        if topology == "2D_mesh":
            self.num_rows, self.num_cols = dimensions
            if pe_count is None:
                self.pe_count = self.num_rows * self.num_cols
            else:
                self.pe_count = pe_count
                if self.pe_count != self.num_rows * self.num_cols:
                    print("Warning: pe_count does not match 2D mesh dimensions.  This might lead to unexpected behavior.")

            self.PEs = {(row, col): PE(pe_id=(row, col), buffer_size=buffer_size)
                        for row in range(self.num_rows)
                        for col in range(self.num_cols)}


        else:
            raise NotImplementedError(f"Topology '{topology}' not yet supported.")

        self.links = self._create_links()  # Create links based on topology
        self.graph = self._create_graph() #Create a graph for visualization


    def _create_graph(self):
      """Creates a NetworkX graph for visualization."""
      graph = nx.DiGraph()  # Directed graph for NoC
      if self.topology == "2D_mesh":
          for row in range(self.num_rows):
              for col in range(self.num_cols):
                  current_pe_id = (row, col)
                  graph.add_node(current_pe_id, pos=(col, -row))  # Position for plotting
                  # Connect to neighbors (right and down)
                  if col + 1 < self.num_cols:
                      right_neighbor_id = (row, col + 1)
                      graph.add_edge(current_pe_id, right_neighbor_id)
                      graph.add_edge(right_neighbor_id, current_pe_id)  # Bidirectional
                  if row + 1 < self.num_rows:
                      down_neighbor_id = (row + 1, col)
                      graph.add_edge(current_pe_id, down_neighbor_id)
                      graph.add_edge(down_neighbor_id, current_pe_id)  # Bidirectional
      return graph

    def visualize_noc(self, fault_locations=None, active_links=None, title="NoC Topology"):
      """
      Visualizes the NoC topology using Matplotlib.

      Args:
          fault_locations (list): List of (src, dst) tuples for faulty links.
          active_links(list) : List of (src, dst) tuples for active links.
      """

      pos = nx.get_node_attributes(self.graph, 'pos')
      plt.figure(figsize=(8, 6))

      # Draw nodes
      nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', node_size=500)
      nx.draw_networkx_labels(self.graph, pos)


      # Draw edges (links)
      #Regular Links
      nx.draw_networkx_edges(self.graph, pos, edgelist=self.links, edge_color='gray')

      #Faulty Links
      if fault_locations:
          nx.draw_networkx_edges(self.graph, pos, edgelist=fault_locations, edge_color='red', width=2)

      #Active Links
      if active_links:
        nx.draw_networkx_edges(self.graph, pos, edgelist=active_links, edge_color='green', width = 2)


      plt.title(title)
      plt.axis('off')  # Turn off axis
      plt.show()


    def _create_links(self):
        """Creates links between PEs based on the topology."""
        links = {}  # Key: (src_pe_id, dst_pe_id), Value: Link object
        if self.topology == "2D_mesh":
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    current_pe_id = (row, col)
                    # Connect to neighbors (right and down)
                    if col + 1 < self.num_cols:
                        right_neighbor_id = (row, col + 1)
                        links[(current_pe_id, right_neighbor_id)] = Link()
                        links[(right_neighbor_id, current_pe_id)] = Link()  # Bidirectional
                    if row + 1 < self.num_rows:
                        down_neighbor_id = (row + 1, col)
                        links[(current_pe_id, down_neighbor_id)] = Link()
                        links[(down_neighbor_id, current_pe_id)] = Link()  # Bidirectional
        return links

    def _xy_routing(self, src_pe_id, dst_pe_id):
        """
        XY routing algorithm for 2D mesh.
        """
        if src_pe_id == dst_pe_id:
            return []  # Already at the destination

        path = [src_pe_id]
        current_pe_id = src_pe_id

        while current_pe_id != dst_pe_id:
            src_row, src_col = current_pe_id
            dst_row, dst_col = dst_pe_id

            if dst_col > src_col and (current_pe_id, (src_row, src_col + 1)) in self.links:
              if self.links[(current_pe_id, (src_row, src_col+1))].is_faulty:
                if not self.fault_tolerance_enabled:
                  return [] #Fail if fault tolerance disabled
                else: #Fault Tolerant
                  next_pe_id = self._alt_route(current_pe_id, dst_pe_id) #Find alternate route.
                  if not next_pe_id:
                      return []  # No alternate route
              else:
                next_pe_id = (src_row, src_col + 1)

            elif dst_col < src_col and (current_pe_id, (src_row, src_col - 1)) in self.links:
              if self.links[(current_pe_id, (src_row, src_col - 1))].is_faulty:
                if not self.fault_tolerance_enabled:
                    return []
                else:
                    next_pe_id = self._alt_route(current_pe_id, dst_pe_id)
                    if not next_pe_id:
                        return []
              else:
                next_pe_id = (src_row, src_col - 1)

            elif dst_row > src_row and (current_pe_id, (src_row + 1, src_col)) in self.links:
              if self.links[(current_pe_id, (src_row + 1, src_col))].is_faulty:
                if not self.fault_tolerance_enabled:
                    return []
                else:
                    next_pe_id = self._alt_route(current_pe_id, dst_pe_id)
                    if not next_pe_id:
                        return []
              else:
                 next_pe_id = (src_row + 1, src_col)


            elif dst_row < src_row and (current_pe_id, (src_row - 1, src_col)) in self.links:
              if self.links[(current_pe_id, (src_row-1, src_col))].is_faulty:
                if not self.fault_tolerance_enabled:
                  return []
                else:
                    next_pe_id = self._alt_route(current_pe_id, dst_pe_id)
                    if not next_pe_id:
                        return []
              else:
                next_pe_id = (src_row - 1, src_col)

            else:
                return []  # No valid path found (should not happen in a healthy 2D mesh)

            path.append(next_pe_id)
            current_pe_id = next_pe_id

        return path

    def _alt_route(self, src_pe_id, dst_pe_id):
        """
        Provides an alternate route in the presence of faults (basic implementation).
        """
        src_row, src_col = src_pe_id
        dst_row, dst_col = dst_pe_id

        possible_next_hops = []

        # Explore possible moves (up, down, left, right)
        if src_row > 0 and not self.links[(src_pe_id, (src_row - 1, src_col))].is_faulty:
            possible_next_hops.append((src_row - 1, src_col))
        if src_row < self.num_rows - 1 and not self.links[(src_pe_id, (src_row + 1, src_col))].is_faulty:
            possible_next_hops.append((src_row + 1, src_col))
        if src_col > 0 and not self.links[(src_pe_id, (src_row, src_col - 1))].is_faulty:
            possible_next_hops.append((src_row, src_col - 1))
        if src_col < self.num_cols - 1 and not self.links[(src_pe_id, (src_row, src_col + 1))].is_faulty:
            possible_next_hops.append((src_row, src_col + 1))


        # Choose the next hop that brings us closer to the destination.
        best_next_hop = None
        min_dist = float('inf')

        for next_hop in possible_next_hops:
            dist = abs(next_hop[0] - dst_row) + abs(next_hop[1] - dst_col)
            if dist < min_dist:
                min_dist = dist
                best_next_hop = next_hop

        return best_next_hop

    def route(self, src_pe_id, dst_pe_id):
        """
        Routes a packet from source to destination PE.
        """
        if self.routing_algo == "XY":
            return self._xy_routing(src_pe_id, dst_pe_id)
        else:
            raise NotImplementedError(f"Routing algorithm '{self.routing_algo}' not supported.")

    def send_packet(self, src_pe_id, dst_pe_id, packet):
        """Simulates sending a packet through the NoC."""

        path = self.route(src_pe_id, dst_pe_id)
        if not path:
            return False, 0, []  # No path, delivery failed.

        current_pe_id = src_pe_id
        latency = 0
        
        #Track active links for visualization
        active_links = []

        for next_pe_id in path[1:]:  # Iterate through the path (excluding the starting PE)
            link = self.links[(current_pe_id, next_pe_id)]
            active_links.append((current_pe_id, next_pe_id))

            # Simulate transmission and error detection.
            if link.transmit(packet):
                latency += link.latency #Add link latency
                if self.fault_tolerance_enabled:
                  if self._detect_error(packet):
                      print(f"Error detected in packet from {current_pe_id} to {next_pe_id}!")
                      if not self._recover_from_error(current_pe_id, next_pe_id, packet, link):
                          print("Error Recovery Failed")
                          return False, latency, [] #Recovery Failed
                      else:
                          print("Error Recovered. Continuing.")
                          #Update latency, after recovery.
                          latency = latency - link.latency + link.get_recovery_latency()

                # Packet arrived at the next PE.  Receive it.
                if not self.PEs[next_pe_id].receive(packet, link):
                  return False, latency, [] #Receive failed
            else:
                print(f"Link failure between {current_pe_id} and {next_pe_id}.")
                return False, latency, [] #Link Failed

            current_pe_id = next_pe_id  # Move to the next PE

        return True, latency, active_links

    def _detect_error(self, packet):
        """Detects errors in a received packet."""
        if not self.fault_tolerance_enabled:
            return False

        if self.error_detection_method == "EDC":
            # Basic parity check (example EDC).  Could be CRC, checksum, etc.
            data = packet.get('data', 0)  # Get data, default to 0 if not present.
            parity_bit = packet.get('parity', 0)

            # Handle NumPy arrays and scalars consistently
            if isinstance(data, np.ndarray):
                calculated_parity = np.bitwise_xor.reduce(data.astype(np.int8).flat)  # Use NumPy for array
            else:
                calculated_parity = 0
                for bit in bin(data)[2:]:  # Count bits in data.
                    if bit == '1':
                        calculated_parity = 1 - calculated_parity  # XOR

            return calculated_parity != parity_bit #Return if error was detected or not

        if self.error_detection_method == "spatial_redundancy":
            # Basic Triple Modular Redundancy Check
            data_1 = packet.get('data_1', 0)  # Get first data segment
            data_2 = packet.get('data_2', 0)  # Get second data segment
            data_3 = packet.get('data_3', 0)

            # Handle NumPy arrays for spatial redundancy
            if isinstance(data_1, np.ndarray):
                return not (np.array_equal(data_1, data_2) and np.array_equal(data_1, data_3))
            else:
                # Majority Vote (for scalar values)
                if data_1 == data_2:
                    if data_1 == data_3:
                        return False  # No Error
                    else:
                        return True  # Error
                if data_1 == data_3:
                    return True
                if data_2 == data_3:
                    return True
                return True # All different (shouldn't happen with 3)

        if self.error_detection_method == "time_redundancy":
            data_1 = packet.get('data_1', 0)  # Get first data segment
            data_2 = packet.get('data_2', 0)
            if isinstance(data_1, np.ndarray):
                return not np.array_equal(data_1, data_2)
            else:
                if data_1 == data_2:
                    return False
                return True #Error

        if self.error_detection_method == "heartbeat":
            #Implementation handled in send_packet.
            return False #We are checking for link errors, not data errors.

        # Add other error detection methods (spatial redundancy, etc.) as needed.
        raise NotImplementedError(f"Error detection method '{self.error_detection_method}' not implemented.")


    def _recover_from_error(self, src_pe_id, dst_pe_id, packet, link):
        """Attempts to recover from a detected error."""
        if not self.fault_tolerance_enabled:
            return False

        if self.error_recovery_method == "retransmission":
            print(f"Retransmitting packet from {src_pe_id} to {dst_pe_id}.")
            #Resend, but mark the link with the delay
            return link.transmit(packet, retransmit=True) #Retransmit, and increment the link retransmission count.

        elif self.error_recovery_method == "rerouting":
            print(f"Rerouting packet from {src_pe_id} due to error.")
            # Mark the link as faulty
            link.is_faulty = True
            # Try a different route
            new_path = self.route(src_pe_id, dst_pe_id)
            if not new_path:
              link.is_faulty = False #Undo the change to the link state, in case it was transient, and there's no other path
              print("No alternate route found, Retransmitting")
              return link.transmit(packet, retransmit=True)  # Attempt retransmission on original link

            link.is_faulty = False #Undo the change, since we use the link object to route
            # Send the packet along the new path, this will add latency!
            new_src = src_pe_id
            for next_hop in new_path[1:]:
              next_link = self.links[(new_src,next_hop)]
              if not next_link.transmit(packet):
                print("New Path Failed")
                return False #Even the new path failed
              new_src = next_hop
            return True

        elif self.error_recovery_method == "checkpointing":
            print("Checkpointing not implemented yet, retransmitting instead")
            return link.transmit(packet) #Retransmit

        elif self.error_recovery_method == "fault_tolerant_routing":
          #This is handled in the route method already.
          return True

        # Implement other error recovery methods (rerouting, checkpointing, etc.)
        raise NotImplementedError(f"Error recovery method '{self.error_recovery_method}' not implemented.")

    def inject_fault(self, src_pe_id, dst_pe_id):
      """Injects a fault into a link"""
      if (src_pe_id, dst_pe_id) in self.links:
        self.links[(src_pe_id, dst_pe_id)].is_faulty = True
      elif (dst_pe_id, src_pe_id) in self.links:
        self.links[(dst_pe_id, src_pe_id)].is_faulty = True
      else:
        print("Link not found")
    
    def remove_fault(self, src_pe_id, dst_pe_id):
        """Removes a fault from a link"""
        if (src_pe_id, dst_pe_id) in self.links:
          self.links[(src_pe_id, dst_pe_id)].is_faulty = False
        elif (dst_pe_id, src_pe_id) in self.links:
          self.links[(dst_pe_id, src_pe_id)].is_faulty = False
        else:
          print("Link not found")

class PE:
    """Represents a Processing Element (PE)."""

    def __init__(self, pe_id, buffer_size):
        """
        Initializes the PE.
        """
        self.pe_id = pe_id
        self.buffer_size = buffer_size
        self.input_buffer = deque(maxlen=buffer_size)  # FIFO buffer
        self.output_buffer = deque(maxlen=buffer_size)
        self.memory = {}  # Can store weights, activations, etc.

    def receive(self, packet, link):
        """
        Receives a packet into the input buffer.
        """
        if len(self.input_buffer) < self.buffer_size:
            self.input_buffer.append(packet)
            link.clear_packet() #Clear the link for the next packet
            return True #Return success
        else:
            # Buffer full, packet dropped.
            print(f"PE {self.pe_id} input buffer full. Packet dropped.")
            return False

    def process(self):
        """Simulates processing of data within the PE."""
        # Placeholder for processing logic (e.g., MAC operations)
        if self.input_buffer:
          packet = self.input_buffer.popleft() #Get from buffer
          #Simulate processing
          if "data" in packet:
              #Very simple operation to show processing.  Could be a MAC or other.
              processed_data = packet['data'] * 2  # Example: double the data
              # Put result in output buffer or memory.
              self.output_buffer.append({'result': processed_data})
          return packet #return the processed data

        return None


class Link:
    """Represents a communication link between two PEs."""

    def __init__(self, bandwidth=1, latency=1):
        """
        Initializes a Link.
        """
        self.bandwidth = bandwidth
        self.latency = latency
        self.is_faulty = False
        self.packet = None #Current Packet
        self.retransmission_count = 0
        self.MAX_RETRANSMISSIONS = 3 #Maximum number of retransmissions before link failure

    def transmit(self, packet, retransmit = False):
        """
        Transmits a packet over the link.
        """
        if self.is_faulty and not retransmit:
            return False #Cannot transmit over faulty link

        if retransmit:
            self.retransmission_count += 1
            if self.retransmission_count > self.MAX_RETRANSMISSIONS:
                print(f"Max Retransmissions reached on link")
                self.is_faulty = True #Mark as permanently faulty
                return False #Fail after too many retransmits

        self.packet = packet #Put packet on the link
        return True #Return success

    def clear_packet(self):
        """Clears the link for the next transmission."""
        self.packet = None
        self.retransmission_count = 0 #Reset after final delivery.

    def get_recovery_latency(self):
      """Return the total latency from retransmissions/rerouting"""
      return self.latency * (self.retransmission_count+1)  # +1 for initial.


class NeuralNetwork:
    """Represents a Neural Network model."""

    def __init__(self, layers):
        """
        Initializes the Neural Network.
        """
        self.layers = layers
        self.layer_mapping = {}  # Store the mapping of layers to PEs

    def create_mapping(self, noc, mapping_strategy = "data_parallel", model_parallel_config = None):
      """
      Creates mapping of layers to PEs based on the chosen strategy.
      """
      self.layer_mapping = {}

      if mapping_strategy == "data_parallel":
          for i, layer in enumerate(self.layers):
              # Assign all PEs to each layer for data parallelism
              self.layer_mapping[i] = set(noc.PEs.keys())

      elif mapping_strategy == "model_parallel":
            if model_parallel_config is None or len(model_parallel_config) != len(self.layers):
                raise ValueError("model_parallel_config must be a list of lists with the same length as the number of layers.")

            for i, layer in enumerate(self.layers):
                self.layer_mapping[i] = set(model_parallel_config[i])
                if not self.layer_mapping[i].issubset(noc.PEs.keys()):
                  raise ValueError(f"Invalid PE IDs in model_parallel_config for layer {i}")


      elif mapping_strategy == "hybrid":
          #Simple Hybrid approach.
          if model_parallel_config is None:
            raise ValueError("model_parallel_config must be provided for hybrid mapping")
          if len(model_parallel_config) != len(self.layers):
            raise ValueError("For this hybrid example, model_parallel_config must have entries for each layer, specifying which PEs are used for THAT LAYER.")
          for i, layer in enumerate(self.layers):
            self.layer_mapping[i] = set(model_parallel_config[i]) #Use provided config, like model_parallel
            if not self.layer_mapping[i].issubset(noc.PEs.keys()):
                raise ValueError(f"Invalid PE IDs in model_parallel_config for layer {i}")

      else:
          raise ValueError(f"Invalid mapping strategy: {mapping_strategy}")

      return self.layer_mapping


    def split_data(self, data, num_splits):
        """
        Splits data for data parallelism.
        """

        if len(data) < num_splits:  # Check if splitting is possible
            raise ValueError("Cannot split data into more parts than there are elements.")

        split_size = len(data) // num_splits  # Calculate the size of each split
        remainder = len(data) % num_splits   # Calculate the remainder

        sub_tensors = []
        start = 0
        for i in range(num_splits):
            end = start + split_size
            if i < remainder:
                end += 1  # Distribute the remainder among the first few splits
            sub_tensors.append(data[start:end])  # Corrected slicing here
            start = end

        return sub_tensors

    def merge_data(self, sub_tensors):
      """Recombine split data (inverse of split)"""
      return np.concatenate(sub_tensors)


class Layer:
    """Represents a layer in the neural network."""

    def __init__(self, layer_type, num_neurons, weights=None, activation_fn="relu"):
        """
        Initializes a Layer.
        """
        self.layer_type = layer_type
        self.num_neurons = num_neurons
        self.activation_fn = activation_fn

        if weights is None:
            # Initialize weights randomly (simplified)
            if layer_type == "dense":
                self.weights = np.random.rand(num_neurons, num_neurons)  # Example for a dense layer
            # Add initialization for other layer types (conv2d, etc.) as needed
        else:
            self.weights = weights



# --- Simulation Functions ---

def run_simulation(noc, nn, input_data, mapping_strategy="data_parallel", model_parallel_config=None, verbose=False):
    """
    Runs a simulation, and visualizes the NoC
    """

    layer_mapping = nn.create_mapping(noc, mapping_strategy, model_parallel_config)  # Create the mapping
    total_latency = 0
    current_data = input_data
    per_layer_latencies = []
    all_layer_outputs = []
    all_active_links = [] #For visualization

    #Visualize initial NoC state
    noc.visualize_noc(title="Initial NoC State")

    for layer_idx, layer in enumerate(nn.layers):
        if verbose: print(f"Processing layer {layer_idx} ({layer.layer_type})")
        assigned_pes = layer_mapping[layer_idx]  # Get the PEs assigned to this layer
        num_pes_assigned = len(assigned_pes)

        # --- Data Parallelism within the Layer ---
        if mapping_strategy == "data_parallel" or mapping_strategy == "hybrid":
            #Split the data for this layer's processing
            data_splits = nn.split_data(current_data, num_pes_assigned)

            # Distribute data and weights to PEs
            pe_idx = 0
            for pe_id in assigned_pes:
              #Put data onto the assigned PE's input.
              noc.PEs[pe_id].input_buffer.append({'data': data_splits[pe_idx], 'layer': layer_idx})
              #Simplified weight handling
              noc.PEs[pe_id].memory['weights'] = layer.weights  # Simplified: each PE gets all weights (for data parallelism)
              pe_idx += 1


            # Simulate PE processing (in parallel - using a loop for simulation)
            layer_outputs = []
            for pe_id in assigned_pes:
                output = noc.PEs[pe_id].process()
                if output is not None:  # If the PE produced an output
                    layer_outputs.append(output['data'])


            #Merge the outputs to pass to the next stage.
            current_data = nn.merge_data(layer_outputs)
            all_layer_outputs.append(current_data)  # Store for later use, if needed.
            per_layer_latencies.append(1) #Simplified, 1 cycle since all PEs operate in parallel

        # --- Model Parallelism (across layers) ---
        elif mapping_strategy == "model_parallel":
            #For Model Parallel, we assume the data is already where it needs to be.
            pe_id = list(assigned_pes)[0] #Model parallel, so we know we have only 1 PE per layer in this case.

            noc.PEs[pe_id].input_buffer.append({'data': current_data, 'layer': layer_idx})
            noc.PEs[pe_id].memory['weights'] = layer.weights #Give the PE the weights

            output = noc.PEs[pe_id].process()
            if output is not None:
                current_data = output['data']
                all_layer_outputs.append(current_data)

            per_layer_latencies.append(1) #1 cycle for processing in this simplification.



        # --- Communication between Layers (Inter-layer communication) ---
        if layer_idx < len(nn.layers) - 1:  # No communication needed after the last layer
            next_layer_pes = layer_mapping[layer_idx + 1]
            max_communication_latency = 0

            #Visualize Communication


            if mapping_strategy == "data_parallel":
              #Data parallel is simple, since all PEs need all the data.
              src_pe = list(assigned_pes)[0] #Pick the first PE

              for dst_pe in next_layer_pes:
                if src_pe != dst_pe:  # Avoid sending to self
                    packet = {'data': current_data, 'layer': layer_idx + 1,
                              'source_layer': layer_idx, 'packet_id' : 0}

                    # Add Error Detection Code
                    if noc.fault_tolerance_enabled:
                        if noc.error_detection_method == "EDC":
                            data = packet['data']
                            # Handle NumPy arrays for EDC
                            if isinstance(data, np.ndarray):
                                calculated_parity = np.bitwise_xor.reduce(data.astype(np.int8).flat)
                            else:
                                calculated_parity = 0
                                for bit in bin(data)[2:]:  # Compute parity bit
                                    if bit == '1':
                                        calculated_parity = 1 - calculated_parity  # XOR
                            packet['parity'] = calculated_parity

                        if noc.error_detection_method == "spatial_redundancy":
                          packet['data_1'] = packet['data']
                          packet['data_2'] = packet['data']
                          packet['data_3'] = packet['data']

                        if noc.error_detection_method == "time_redundancy":
                            packet['data_1'] = packet['data']
                            packet['data_2'] = packet['data']


                    success, latency, active_links = noc.send_packet(src_pe, dst_pe, packet)
                    all_active_links.extend(active_links)

                    if success:
                        if verbose: print(f"  Sent data from layer {layer_idx} PE {src_pe} to layer {layer_idx+1} PE {dst_pe} (latency: {latency})")
                        max_communication_latency = max(max_communication_latency, latency) #Keep track of the max latency

                    else:
                         print(f"Communication failed from layer {layer_idx} to layer {layer_idx + 1}")
                         # In a real system, you might have more sophisticated error handling here

            if mapping_strategy == "model_parallel" or mapping_strategy == 'hybrid':

              src_pes = list(assigned_pes)
              dst_pes = list(next_layer_pes)

              for i in range(len(src_pes)):  # Iterate and send between corresponding PEs
                src_pe = src_pes[i]
                dst_pe = dst_pes[i] #Simple 1-1 mapping
                packet = {'data': current_data, 'layer': layer_idx + 1,
                              'source_layer': layer_idx, 'packet_id' : 0}

                # Add Error Detection Code
                if noc.fault_tolerance_enabled:
                    if noc.error_detection_method == "EDC":
                        data = packet['data']
                        # Handle NumPy arrays for EDC
                        if isinstance(data, np.ndarray):
                            calculated_parity = np.bitwise_xor.reduce(data.astype(np.int8).flat)
                        else:
                            calculated_parity = 0
                            for bit in bin(data)[2:]:  # Compute parity bit
                                if bit == '1':
                                    calculated_parity = 1 - calculated_parity  # XOR
                        packet['parity'] = calculated_parity

                    if noc.error_detection_method == "spatial_redundancy":
                        packet['data_1'] = packet['data']
                        packet['data_2'] = packet['data']
                        packet['data_3'] = packet['data']

                    if noc.error_detection_method == "time_redundancy":
                        packet['data_1'] = packet['data']
                        packet['data_2'] = packet['data']

                success, latency, active_links = noc.send_packet(src_pe, dst_pe, packet)
                all_active_links.extend(active_links)

                if success:
                    if verbose: print(                        f"  Sent data from layer {layer_idx} PE {src_pe} to layer {layer_idx + 1} PE {dst_pe} (latency: {latency})")
                    max_communication_latency = max(max_communication_latency, latency)
                else:
                     print(f"Communication failed from layer {layer_idx} to layer {layer_idx + 1}")
            total_latency += max_communication_latency
            per_layer_latencies[-1] += max_communication_latency #Add communication latency to the PREVIOUS layer.

        #Visualize after each layer
        faults = [link for link in noc.links if noc.links[link].is_faulty] #Get faulty links
        noc.visualize_noc(fault_locations=faults, active_links = all_active_links,
                          title = f"NoC State after Layer {layer_idx}")
        all_active_links = [] #Clear for next layer


    return total_latency, current_data, per_layer_latencies


# --- Example Usage and Case Studies ---

def case_study_data_parallelism():
    """Demonstrates data parallelism and basic NoC functionality."""
    print("-" * 20, "Case Study: Data Parallelism", "-" * 20)

    # 1. Define the NoC
    noc = NoC(topology="2D_mesh", dimensions=(4, 4), buffer_size=10)

    # 2. Define the Neural Network (simple two-layer network)
    layer1 = Layer(layer_type="dense", num_neurons=16)  # Fully connected layer
    layer2 = Layer(layer_type="dense", num_neurons=16)
    nn = NeuralNetwork(layers=[layer1, layer2])

    # 3. Input Data (example)
    input_data = np.arange(16)  # Simple input data

    # 4. Run the Simulation
    total_latency, final_output, per_layer_latencies = run_simulation(
        noc, nn, input_data, mapping_strategy="data_parallel", verbose=True
    )

    print("\nSimulation Results:")
    print("  Total Latency:", total_latency)
    print("  Final Output:", final_output)  # Simplified output (last layer's data)
    print("  Per-Layer Latencies:", per_layer_latencies)



def case_study_model_parallelism():
    """Demonstrates model parallelism."""
    print("-" * 20, "Case Study: Model Parallelism", "-" * 20)
    noc = NoC(topology="2D_mesh", dimensions=(4, 4), buffer_size=10)

    # Two layers
    layer1 = Layer(layer_type="dense", num_neurons=64)
    layer2 = Layer(layer_type="dense", num_neurons=32)
    nn = NeuralNetwork(layers=[layer1, layer2])

    input_data = np.arange(64) #Input to first layer.

    # Define which PEs are assigned to each layer.
    model_parallel_config = [
        [(0, 0)],  # Layer 0 assigned to PE (0, 0)
        [(0, 1)],  # Layer 1 assigned to PE (0, 1)
    ]
    total_latency, final_output, per_layer_latencies = run_simulation(
      noc, nn, input_data, mapping_strategy="model_parallel",
      model_parallel_config=model_parallel_config, verbose=True)

    print("\nSimulation Results:")
    print("  Total Latency:", total_latency)
    print("  Final Output:", final_output)
    print("  Per-Layer Latencies:", per_layer_latencies)



def case_study_hybrid_parallelism():
    """Demonstrate hybrid parallelism"""
    print("-" * 20, "Case Study: Hybrid Parallelism", "-" * 20)
    noc = NoC(topology="2D_mesh", dimensions=(4,4), buffer_size = 10)
    layer1 = Layer(layer_type="dense", num_neurons=16)
    layer2 = Layer(layer_type="dense", num_neurons=16)
    nn = NeuralNetwork(layers=[layer1, layer2])

    input_data = np.arange(16)  # Input to the first layer

    #Hybrid config
    model_parallel_config = [
        [(0, 0), (0, 1), (1, 0), (1, 1)],  # Layer 0 on 4 PEs
        [(2, 0), (2, 1), (3, 0), (3, 1)],  # Layer 1 on 4 different PEs
    ]
    total_latency, final_output, per_layer_latencies = run_simulation(
        noc, nn, input_data, mapping_strategy="hybrid",
        model_parallel_config=model_parallel_config, verbose = True
    )

    print("\nSimulation Results:")
    print("  Total Latency:", total_latency)
    print("  Final Output:", final_output)
    print("  Per-Layer Latencies:", per_layer_latencies)


def case_study_fault_tolerance():
  """Demonstrate fault injection and handling"""
  print("-" * 20, "Case Study: Fault Tolerance", "-" * 20)
  #1. Define NoC
  noc = NoC(topology="2D_mesh", dimensions=(4, 4), buffer_size=10,
            fault_tolerance_enabled=True, error_detection_method="EDC",
            error_recovery_method="rerouting")
  #2. Define the neural network
  layer1 = Layer(layer_type="dense", num_neurons=16)  # Fully connected layer
  layer2 = Layer(layer_type="dense", num_neurons=16)
  nn = NeuralNetwork(layers=[layer1, layer2])

  # 3. Input Data (example)
  input_data = np.arange(16)  # Simple input data

  #Inject a fault BEFORE simulation.
  noc.inject_fault((0,0), (0,1))  # Inject a fault into a specific link.

  # 4. Run the Simulation
  total_latency, final_output, per_layer_latencies = run_simulation(
    noc, nn, input_data, mapping_strategy="data_parallel", verbose=True
  )
  noc.remove_fault((0,0),(0,1)) #Remove the fault for next run
  print("\nSimulation Results (with fault injection):")
  print("  Total Latency:", total_latency)
  print("  Final Output:", final_output)
  print("  Per-Layer Latencies:", per_layer_latencies)

  # Run again, *without* the fault, to compare.
  total_latency_nofault, final_output_nofault, per_layer_latencies_nofault = run_simulation(
    noc, nn, input_data, mapping_strategy="data_parallel", verbose=False
  )
  print("\nSimulation Results (NO fault):")
  print("  Total Latency:", total_latency_nofault)
  print("  Final Output:", final_output_nofault)
  print("  Per-Layer Latencies:", per_layer_latencies_nofault)

  print("\nComparison:")
  print("  Latency Increase due to Fault:", total_latency - total_latency_nofault)




def case_study_different_NoC_sizes():
    """Analyze impact of different NoC sizes"""
    print("-" * 20, "Case Study: Different NoC Sizes", "-" * 20)

    noc_sizes = [(2, 2), (4, 4), (8, 8)]
    layer1 = Layer(layer_type="dense", num_neurons=64)
    layer2 = Layer(layer_type="dense", num_neurons=64)
    nn = NeuralNetwork(layers=[layer1, layer2])
    input_data = np.arange(64)


    for size in noc_sizes:
        noc = NoC(topology="2D_mesh", dimensions=size, buffer_size=16)
        total_latency, _, per_layer_latencies = run_simulation(noc, nn, input_data, mapping_strategy="data_parallel")

        print(f"\nNoC Size: {size}")
        print("  Total Latency:", total_latency)
        #print("  Per-Layer Latencies:", per_layer_latencies)


def case_study_error_methods():
    """Compares different error detection/recovery methods."""
    print("-" * 20, "Case Study: Different Error Methods", "-" * 20)

    error_configs = [
        ("EDC", "retransmission"),
        ("spatial_redundancy", "rerouting"),
        ("time_redundancy", "retransmission"),
    ]

    # Same network and input for all configurations.
    noc = NoC(topology="2D_mesh", dimensions=(4, 4), buffer_size=10, fault_tolerance_enabled=True)
    layer1 = Layer(layer_type="dense", num_neurons=16)
    layer2 = Layer(layer_type="dense", num_neurons=16)
    nn = NeuralNetwork(layers=[layer1, layer2])
    input_data = np.arange(16)

    for detection_method, recovery_method in error_configs:
        print(f"\nTesting Error Detection: {detection_method}, Recovery: {recovery_method}")
        noc.error_detection_method = detection_method
        noc.error_recovery_method = recovery_method

        # Inject a fault
        noc.inject_fault((0, 0), (0, 1))

        total_latency, _, _ = run_simulation(noc, nn, input_data, mapping_strategy="data_parallel")
        print("  Total Latency (with fault and recovery):", total_latency)

        # Remove the fault for the next run
        noc.remove_fault((0, 0), (0, 1))


# --- Main Execution ---
if __name__ == "__main__":
    # Run the case studies to demonstrate model utility
    case_study_data_parallelism()
    case_study_model_parallelism()
    case_study_hybrid_parallelism()
    case_study_fault_tolerance()
    case_study_different_NoC_sizes()
    case_study_error_methods()
