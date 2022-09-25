# cython: language_level=3

cimport numpy as np
from numpy import pi


cdef float[31] pulse_table = [
    0.0,
    0.011609139523578026,
    0.022939481268011527,
    0.03400094921689606,
    0.04480300187617261,
    0.05535465924895688,
    0.06566452795600367,
    0.07574082464884459,
    0.08559139784946236,
    0.09522374833850243,
    0.10464504820333041,
    0.11386215864759427,
    0.12288164665523155,
    0.13170980059397538,
    0.14035264483627205,
    0.1488159534690486,
    0.15710526315789472,
    0.16522588522588522,
    0.1731829170024174,
    0.18098125249301955,
    0.18862559241706162,
    0.19612045365662886,
    0.20347017815646784,
    0.21067894131185272,
    0.21775075987841944,
    0.2246894994354535,
    0.2314988814317673,
    0.23818248984115256,
    0.2447437774524158,
    0.2511860718171926,
    0.25751258087706685,
]

cdef float[202] tnd_table = [
    0.0,
    0.006699823979696262,
    0.01334502018019487,
    0.01993625400950099,
    0.026474180112418616,
    0.032959442587297105,
    0.03939267519756107,
    0.04577450157816932,
    0.05210553543714433,
    0.05838638075230885,
    0.06461763196336215,
    0.07079987415942428,
    0.07693368326217241,
    0.08301962620468999,
    0.08905826110614481,
    0.09505013744240969,
    0.10099579621273477,
    0.10689577010257789,
    0.11275058364269584,
    0.11856075336459644,
    0.12432678795244785,
    0.1300491883915396,
    0.13572844811338536,
    0.1413650531375568,
    0.1469594822103333,
    0.15251220694025122,
    0.15802369193063237,
    0.16349439490917161,
    0.16892476685465738,
    0.1743152521209005,
    0.1796662885579421,
    0.18497830763060993,
    0.19025173453449087,
    0.19548698830938505,
    0.20068448195030472,
    0.20584462251608032,
    0.2109678112356332,
    0.2160544436119733,
    0.2211049095239788,
    0.22611959332601225,
    0.2310988739454269,
    0.23604312497801538,
    0.24095271478145042,
    0.24582800656676793,
    0.25066935848793903,
    0.25547712372957787,
    0.2602516505928307,
    0.26499328257948945,
    0.26970235847437257,
    0.27437921242601526,
    0.27902417402570834,
    0.28363756838492643,
    0.2882197162111822,
    0.292770933882345,
    0.29729153351945914,
    0.3017818230580978,
    0.3062421063182866,
    0.31067268307302937,
    0.31507384911547015,
    0.3194458963247213,
    0.32378911273039,
    0.3281037825758322,
    0.3323901863801631,
    0.33664860099905314,
    0.3408792996843372,
    0.34508255214246325,
    0.349258624591807,
    0.3534077798188791,
    0.3575302772334479,
    0.36162637292260397,
    0.3656963197037888,
    0.3697403671768112,
    0.3737587617748739,
    0.37775174681463214,
    0.38171956254530554,
    0.38566244619686446,
    0.3895806320273106,
    0.3934743513690717,
    0.3973438326745308,
    0.40118930156070615,
    0.405010980853104,
    0.4088090906287582,
    0.41258384825847705,
    0.4163354684483128,
    0.42006416328027124,
    0.4237701422522769,
    0.42745361231741014,
    0.4311147779224318,
    0.4347538410456096,
    0.43837100123386197,
    0.4419664556392331,
    0.44554039905471293,
    0.44909302394941686,
    0.4526245205031371,
    0.45613507664027986,
    0.4596248780632002,
    0.4630941082849479,
    0.4665429486614358,
    0.46997157842304194,
    0.47338017470565896,
    0.4767689125811996,
    0.48013796508757145,
    0.48348750325813084,
    0.48681769615062515,
    0.49012871087563703,
    0.493420712624537,
    0.49669386469695664,
    0.49994832852779125,
    0.5031842637137408,
    0.5064018280393993,
    0.5096011775029012,
    0.5127824663411329,
    0.5159458470545188,
    0.5190914704313901,
    0.5222194855719443,
    0.5253300399118033,
    0.528423279245178,
    0.5314993477476477,
    0.5345583879985607,
    0.5376005410030638,
    0.5406259462137686,
    0.5436347415520602,
    0.5466270634290563,
    0.5496030467662235,
    0.5525628250156552,
    0.5555065301800212,
    0.5584342928321915,
    0.5613462421345432,
    0.5642425058579547,
    0.5671232104004943,
    0.5699884808058077,
    0.5728384407812124,
    0.5756732127155,
    0.5784929176964575,
    0.5812976755281083,
    0.5840876047476803,
    0.5868628226423054,
    0.5896234452654553,
    0.5923695874531196,
    0.595101362839729,
    0.5978188838738291,
    0.6005222618335111,
    0.6032116068415997,
    0.6058870278806079,
    0.6085486328074569,
    0.6111965283679723,
    0.6138308202111536,
    0.6164516129032258,
    0.6190590099414757,
    0.6216531137678758,
    0.6242340257825014,
    0.6268018463567424,
    0.6293566748463153,
    0.6318986096040777,
    0.6344277479926501,
    0.6369441863968464,
    0.6394480202359187,
    0.6419393439756177,
    0.6444182511400732,
    0.6468848343234979,
    0.6493391852017159,
    0.6517813945435207,
    0.6542115522218658,
    0.6566297472248885,
    0.659036067666773,
    0.6614306007984521,
    0.6638134330181533,
    0.6661846498817908,
    0.6685443361132047,
    0.670892575614252,
    0.6732294514747513,
    0.6755550459822829,
    0.6778694406318475,
    0.6801727161353863,
    0.6824649524311629,
    0.684746228693012,
    0.6870166233394548,
    0.6892762140426848,
    0.6915250777374256,
    0.693763290629662,
    0.6959909282052493,
    0.6982080652383982,
    0.7004147758000423,
    0.7026111332660865,
    0.70479721032554,
    0.7069730789885358,
    0.7091388105942369,
    0.7112944758186339,
    0.7134401446822323,
    0.7155758865576349,
    0.7177017701770176,
    0.7198178636395035,
    0.7219242344184336,
    0.7240209493685391,
    0.7261080747330146,
    0.7281856761504939,
    0.7302538186619317,
    0.7323125667173908,
    0.734361984182737,
    0.7364021343462434,
    0.7384330799251054,
    0.7404548830718675,
 ]



def apu_mixer(
    apu,
    np.ndarray[np.uint8_t, ndim=1] pulse1,
    np.ndarray[np.uint8_t, ndim=1] pulse2,
    np.ndarray[np.uint8_t, ndim=1] triangle,
    np.ndarray[np.uint8_t, ndim=1] noise,
    np.ndarray[np.uint8_t, ndim=1] dmc,
    np.ndarray[np.int16_t, ndim=2] output,
):
    cdef unsigned int ticks_in_frame = pulse1.shape[0]
    cdef unsigned int i
    cdef short value

    cdef float[14890] mixer_out
    cdef float current_in, current_out, alpha, omega
    cdef float dt = 1/(60 * ticks_in_frame)

    cdef float filter1_previous_in = apu.filter1_previous_in
    cdef float filter1_previous_out = apu.filter1_previous_out
    cdef float filter2_previous_in = apu.filter2_previous_in
    cdef float filter2_previous_out = apu.filter2_previous_out
    cdef float filter3_previous_in = apu.filter3_previous_in
    cdef float filter3_previous_out = apu.filter3_previous_out

    # Mixing
    for i in range(ticks_in_frame):
        mixer_out[i] = (
            pulse_table[pulse1[i] + pulse2[i]] +
            tnd_table[3 * triangle[i] + 2 * noise[i] + dmc[i]]
        )

    # High pass filter 1 (90 Hz)
    if apu.filter1_enabled:
        omega = 2.0 * pi * apu.filter1_cutoff
        alpha = 1 / (1 + omega * dt)
        for i in range(1, ticks_in_frame):
            current_in = mixer_out[i]
            current_out = (filter1_previous_out + current_in - filter1_previous_in) * alpha
            mixer_out[i] = current_out
            filter1_previous_in = current_in
            filter1_previous_out = current_out

    # High pass filter 2 (442 Hz)
    if apu.filter2_enabled:
        omega = 2.0 * pi * apu.filter2_cutoff
        alpha = 1 / (1 + omega * dt)
        for i in range(1, ticks_in_frame):
            current_in = mixer_out[i]
            current_out = (filter2_previous_out + current_in - filter2_previous_in) * alpha
            mixer_out[i] = current_out
            filter2_previous_in = current_in
            filter2_previous_out = current_out

    # Low pass filter 3 (14 kHz)
    if apu.filter3_enabled:
        omega = 2.0 * pi * apu.filter3_cutoff
        alpha = (omega * dt) / (1 + omega * dt)
        for i in range(1, ticks_in_frame):
            current_in = mixer_out[i]
            current_out = filter3_previous_out + (current_in - filter3_previous_out) * alpha
            mixer_out[i] = current_out
            filter3_previous_in = current_in
            filter3_previous_out = current_out

    for i in range(ticks_in_frame):
        value = <short>(mixer_out[i] * 32768)
        output[(i<<1)|0,0] = value
        output[(i<<1)|0,1] = value
        output[(i<<1)|1,0] = value
        output[(i<<1)|1,1] = value

    apu.filter1_previous_in = filter1_previous_in
    apu.filter1_previous_out = filter1_previous_out
    apu.filter2_previous_in = filter2_previous_in
    apu.filter2_previous_out = filter2_previous_out
    apu.filter3_previous_in = filter3_previous_in
    apu.filter3_previous_out = filter3_previous_out


def generate_pulse(
    pulse,
    np.ndarray[np.uint8_t, ndim=1] pulse_out,
):
    cdef unsigned int ticks_in_frame = len(pulse_out)
    cdef unsigned int i, value, current_envelope

    # Parameters
    cdef unsigned short duty = pulse.duty
    cdef unsigned short volume = pulse.volume
    cdef unsigned short load_timer = pulse.load_timer
    cdef unsigned short sweep_negate = pulse.sweep_negate
    cdef unsigned short sweep_period = pulse.sweep_period
    cdef unsigned short sweep_enabled = pulse.sweep_enabled
    cdef unsigned short constant_volume = pulse.constant_volume
    cdef unsigned short sweep_reload_flag = pulse.sweep_reload_flag
    cdef unsigned short sweep_shift_count = pulse.sweep_shift_count
    cdef unsigned short load_length_counter = pulse.load_length_counter
    cdef unsigned short length_counter_halt = pulse.length_counter_halt

    # Internal state
    cdef unsigned short enabled = pulse.enabled
    cdef unsigned short start_flag = pulse.start_flag
    cdef unsigned short current_tick = pulse.current_tick
    cdef unsigned short current_timer = pulse.current_timer
    cdef unsigned short sweep_divider = pulse.sweep_divider
    cdef unsigned short length_counter = pulse.length_counter
    cdef unsigned short divider_period = pulse.divider_period
    cdef unsigned short current_sequencer = pulse.current_sequencer
    cdef unsigned short decay_level_counter = pulse.decay_level_counter
    cdef unsigned short current_timer_period = pulse.current_timer_period

    cdef unsigned char[8] sequencer_table

    if duty == 0:
        sequencer_table = [0, 0, 0, 0, 0, 0, 0, 1]
    elif duty == 1:
        sequencer_table = [0, 0, 0, 0, 0, 0, 1, 1]
    elif duty == 2:
        sequencer_table = [0, 0, 0, 0, 1, 1, 1, 1]
    elif duty == 3:
        sequencer_table = [1, 1, 1, 1, 1, 1, 0, 0]
    else:
        assert False

    current_envelope = volume if constant_volume else decay_level_counter
    value = sequencer_table[current_sequencer] * current_envelope
    for i in range(ticks_in_frame):
        # Manage envelope
        if current_tick in (3728, 7456, 11185, 18640):
            if start_flag:
                start_flag = 0
                decay_level_counter = 15
                divider_period = volume
            elif divider_period != 0:
                divider_period -= 1
            else:
                divider_period = volume
                if decay_level_counter != 0:
                    decay_level_counter -= 1
                elif length_counter_halt == 1:
                    decay_level_counter = 15
            current_envelope = volume if constant_volume else decay_level_counter
        # Manage length counter
        if current_tick in (7456, 18640):
            if length_counter_halt == 0 and length_counter != 0:
                length_counter -= 1
            # Perform the sweep
            if sweep_divider == 0 and sweep_enabled and current_timer_period >= 8:
                if sweep_negate:
                    current_timer_period -= (current_timer_period >> sweep_shift_count)
                else:
                    current_timer_period += (current_timer_period >> sweep_shift_count)
            # Update sweep divider
            if sweep_divider == 0 or sweep_reload_flag:
                sweep_reload_flag = 0
                sweep_divider = sweep_period
            else:
                sweep_divider -= 1
        # Manage frame counter
        current_tick += 1
        if current_tick == 18641:
            current_tick = 0
        # Manage timer
        if current_timer != 0:
            current_timer -= 1
        else:
            current_timer = current_timer_period
            current_sequencer = 7 if current_sequencer == 0 else current_sequencer - 1
            value = sequencer_table[current_sequencer] * current_envelope
        # Manage length counter
        if enabled and length_counter > 0 and current_timer_period >= 8:
            pulse_out[i] = value
        else:
            pulse_out[i] = 0

    pulse.start_flag = start_flag
    pulse.current_tick = current_tick
    pulse.current_timer = current_timer
    pulse.sweep_divider = sweep_divider
    pulse.length_counter = length_counter
    pulse.divider_period = divider_period
    pulse.sweep_reload_flag = sweep_reload_flag
    pulse.current_sequencer = current_sequencer
    pulse.decay_level_counter = decay_level_counter
    pulse.current_timer_period = current_timer_period

def generate_triangle(
    triangle,
    np.ndarray[np.uint8_t, ndim=1] triangle_out,
):
    cdef unsigned int ticks_in_frame = len(triangle_out)
    cdef unsigned int i, value

    # Parameters
    cdef unsigned char enabled = triangle.enabled
    cdef unsigned short load_timer = triangle.load_timer
    cdef unsigned short load_counter = triangle.load_counter
    cdef unsigned short load_length_counter = triangle.load_length_counter
    cdef unsigned short length_counter_halt = triangle.length_counter_halt

    # Internal state
    cdef unsigned short current_tick = triangle.current_tick
    cdef unsigned short current_value = triangle.current_value
    cdef unsigned short current_timer = triangle.current_timer
    cdef unsigned short length_counter = triangle.length_counter
    cdef unsigned short current_counter = triangle.current_counter
    cdef unsigned short current_sequencer = triangle.current_sequencer
    cdef unsigned short counter_reload_flag = triangle.counter_reload_flag

    cdef unsigned char[32] sequencer_table = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    ]

    for i in range(ticks_in_frame):
        # Manage volume
        if current_tick in (3728, 7456, 11185, 18640):
            if counter_reload_flag:
                counter_reload_flag = 0
                current_counter = load_counter
            elif current_counter != 0:
                current_counter -= 1
        # Manage length counter
        if current_tick in (7456, 18640):
            if length_counter_halt == 0 and length_counter != 0:
                length_counter -= 1
        # Manage frame counter
        current_tick += 1
        if current_tick == 18641:
            current_tick = 0
        # Manage timer
        if current_timer > 1:
            current_timer -= 2
        else:
            current_timer = load_timer if current_timer == 1 else load_timer - 1
            if current_counter != 0 and load_timer >= 2 and enabled and length_counter != 0:
                current_sequencer = 31 if current_sequencer == 0 else current_sequencer - 1
            current_value = sequencer_table[current_sequencer]
        triangle_out[i] = current_value

    triangle.current_value = current_value
    triangle.current_tick = current_tick
    triangle.current_timer = current_timer
    triangle.length_counter = length_counter
    triangle.current_counter = current_counter
    triangle.current_sequencer = current_sequencer
    triangle.counter_reload_flag = counter_reload_flag


def generate_noise(
    noise,
    np.ndarray[np.uint8_t, ndim=1] noise_out,
):
    cdef unsigned int ticks_in_frame = len(noise_out)
    cdef unsigned int i, current_envelope, shift, feedback

    # Parameters
    cdef unsigned short volume = noise.volume
    cdef unsigned short enabled = noise.enabled
    cdef unsigned short noise_mode = noise.noise_mode
    cdef unsigned short noise_period = noise.noise_period
    cdef unsigned short constant_volume = noise.constant_volume
    cdef unsigned short load_length_counter = noise.load_length_counter
    cdef unsigned short length_counter_halt = noise.length_counter_halt

    # Internal state
    cdef unsigned short start_flag = noise.start_flag
    cdef unsigned short current_tick = noise.current_tick
    cdef unsigned short current_timer = noise.current_timer
    cdef unsigned short length_counter = noise.length_counter
    cdef unsigned short shift_register = noise.shift_register
    cdef unsigned short divider_period = noise.divider_period
    cdef unsigned short decay_level_counter = noise.decay_level_counter

    shift = 6 if noise_mode else 1
    current_envelope = volume if constant_volume else decay_level_counter
    for i in range(ticks_in_frame):
        # Manage envelope
        if current_tick in (3728, 7456, 11185, 18640):
            if start_flag:
                start_flag = 0
                decay_level_counter = 15
                divider_period = volume
            elif divider_period != 0:
                divider_period -= 1
            else:
                divider_period = volume
                if decay_level_counter != 0:
                    decay_level_counter -= 1
                elif length_counter_halt == 1:
                    decay_level_counter = 15
            current_envelope = volume if constant_volume else decay_level_counter
        # Manage length counter
        if current_tick in (7456, 18640):
            if length_counter_halt == 0 and length_counter != 0:
                length_counter -= 1
        # Manage frame counter
        current_tick += 1
        if current_tick == 18641:
            current_tick = 0
        # Manage timer
        if current_timer != 0:
            current_timer -= 1
        else:
            current_timer = noise_period
            feedback = (shift_register ^ (shift_register >> shift)) & 0x01
            shift_register >>= 1
            shift_register |= feedback << 14
        # Manage length counter
        if enabled and (shift_register & 0x01) == 0 and length_counter != 0:
            noise_out[i] = current_envelope
        else:
            noise_out[i] = 0

    noise.start_flag = start_flag
    noise.current_tick = current_tick
    noise.current_timer = current_timer
    noise.length_counter = length_counter
    noise.shift_register = shift_register
    noise.divider_period = divider_period
    noise.decay_level_counter = decay_level_counter