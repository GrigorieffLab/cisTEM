/*
 * scattering_potential.h
 *
 *  Created on: Oct 3, 2019
 *      Author: himesb
 */
#include "../../constants/electron_scattering.h"

#ifndef PROGRAMS_SIMULATE_SCATTERING_POTENTIAL_H_
#define PROGRAMS_SIMULATE_SCATTERING_POTENTIAL_H_

#define MAX_NUMBER_PDBS 64 // This probably doesn't need to be limited - check use TODO.

class ScatteringPotential {

  public:
    ScatteringPotential( );
    virtual ~ScatteringPotential( );

    PDB*     pdb_ensemble;
    wxString pdb_file_names[MAX_NUMBER_PDBS];
    int      number_of_pdbs            = 1;
    bool     is_allocated_pdb_ensemble = false;

    __inline__ float ReturnScatteringParamtersA(AtomType id, int term_number) { return SCATTERING_PARAMETERS_A[id][term_number]; }

    __inline__ float ReturnScatteringParamtersB(AtomType id, int term_number) { return SCATTERING_PARAMETERS_B[id][term_number]; }

    __inline__ float ReturnAtomicNumber(AtomType id) { return ATOMIC_NUMBER[id]; }

    void InitPdbEnsemble(float wanted_pixel_size, bool shift_by_cetner_of_mass, int minimum_padding_x_and_y, int minimum_thickness_z,
                         int               max_number_of_noise_particles,
                         float             wanted_noise_particle_radius_as_mutliple_of_particle_radius,
                         float             wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
                         float             wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
                         float             wanted_tilt_angle_to_emulat,
                         bool              is_alpha_fold_prediction,
                         cisTEMParameters& wanted_star_file, bool use_star_file);
    long ReturnTotalNumberOfNonWaterAtoms( );

  private:
    const float WN = 0.8045 * 0.79; // sum netOxy A / sum water (A) = 0.8045 and ratio of total elastic cross section water/oxygen 0.67-0.92 Using average 0.79 (there is no fixed estimate)
};

#endif /* PROGRAMS_SIMULATE_SCATTERING_POTENTIAL_H_ */
