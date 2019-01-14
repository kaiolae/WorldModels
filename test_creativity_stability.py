FIREBALL_THRESHOLD = 0.5
WALL_THRESHOLD = 0.39
#How many consecutive frames to say we've gone into and out of an explosion.
EXPLOSION_START_THRESHOLD = 3
EXPLOSION_END_THRESHOLD = 3

from count_monsters_and_balls import count_monsters, count_fireballs, is_there_a_big_explosion, is_there_a_wall
import numpy as np

#TODO May also need a method that takes two sequences (real and dreamed) and measures DIFFERENCES.
def count_events_from_images(image_sequence):
    num_fireballs = 0
    num_monsters = 0
    thresholded_images = [] #Potentially useful for debugging
    images_with_explosion = []
    images_without_explosion = []
    currently_exploding = False
    num_initiated_explosions = 0
    current_consecutive_explosion_images_counter = 0
    current_consecutive_non_explosion_images_counter = 0
    for img in image_sequence:
        fb, thresholded_image = count_fireballs(img)
        thresholded_images.append(thresholded_image)

        num_fireballs+=fb

        monsters, thresholded_imgage2 = count_monsters(img)
        is_exploding = is_there_a_big_explosion(img)
        if is_exploding:
            current_consecutive_explosion_images_counter +=1
            current_consecutive_non_explosion_images_counter =0
            images_with_explosion.append(img)
        else:
            current_consecutive_explosion_images_counter = 0
            current_consecutive_non_explosion_images_counter += 1
            images_without_explosion.append(img)
        if not currently_exploding and current_consecutive_explosion_images_counter >= EXPLOSION_START_THRESHOLD:
             currently_exploding = True
             num_initiated_explosions += 1

        if currently_exploding and current_consecutive_non_explosion_images_counter >= EXPLOSION_END_THRESHOLD:
             currently_exploding = False
        num_monsters += monsters

    return {"num_fireballs":num_fireballs, "num_monsters":num_monsters, "with_explosion": images_with_explosion,
            "without_explosion": images_without_explosion, "num_with_explosion": len(images_with_explosion),
            "num_without_explosion": len(images_without_explosion), "num_initiated_explosions": num_initiated_explosions}

def count_appearances_and_disappearances_from_images(image_sequence):
    fireball_delta = 0
    monster_delta = 0
    thresholded_images = [] #Potentially useful for debugging
    for img_counter in range(len(image_sequence)):
        if img_counter==0:
            continue
        fb_after, thresholded_image = count_fireballs(image_sequence[img_counter], FIREBALL_THRESHOLD)
        fb_before, thresholded_image = count_fireballs(image_sequence[img_counter-1], FIREBALL_THRESHOLD)
        thresholded_images.append(thresholded_image)

        fireball_delta+=abs(fb_after-fb_before)

        monsters_after, thresholded_image = count_monsters(image_sequence[img_counter])
        monsters_before, thresholded_image = count_monsters(image_sequence[img_counter-1])
        monster_delta += abs(monsters_after-monsters_before)

    return {"fireball_delta":fireball_delta, "monster_delta":monster_delta}


def count_different_events_in_images(real_images, predicted_images):
    #TODO: Note it's important the caller aligns these, so the prediction for t=0 and real event at t=0 are both at index 0 in arrays.
    assert(len(real_images) == len(predicted_images))
    missing_fireballs = 0
    imagined_fireballs = 0
    missing_monsters = 0
    imagined_monsters = 0
    missing_explosions = 0
    missing_walls = 0
    imagined_explosions = 0
    imagined_walls = 0
    for i in range(len(real_images)):
        actual_num_fireballs, img = count_fireballs(real_images[i])
        predicted_num_fireballs, thresholded_image = count_fireballs(predicted_images[i])

        if actual_num_fireballs>predicted_num_fireballs:
            missing_fireballs+=actual_num_fireballs-predicted_num_fireballs
        elif predicted_num_fireballs>actual_num_fireballs:
            imagined_fireballs+=predicted_num_fireballs-actual_num_fireballs

        actual_num_monsters, img = count_monsters(real_images[i])
        predicted_num_monsters, img = count_monsters(predicted_images[i])
        if actual_num_monsters>predicted_num_monsters:
            missing_monsters+=actual_num_monsters-predicted_num_monsters
        elif predicted_num_monsters>actual_num_monsters:
            imagined_monsters+=predicted_num_monsters-actual_num_monsters

        is_actual_explosion = is_there_a_big_explosion(real_images[i])
        is_predicted_explosion = is_there_a_big_explosion(predicted_images[i])

        if is_actual_explosion and not is_predicted_explosion:
            missing_explosions+=1
        if is_predicted_explosion and not is_actual_explosion:
            imagined_explosions+=1

        is_actual_wall = is_there_a_wall(real_images[i], WALL_THRESHOLD)
        is_predicted_wall = is_there_a_wall(predicted_images[i], WALL_THRESHOLD)

        if is_actual_wall and not is_predicted_wall:
            missing_walls+=1
        if is_predicted_wall and not is_actual_wall:
            imagined_walls+=1

    return {"missing_fireballs": missing_fireballs, "imagined_fireballs": imagined_fireballs, "missing_monsters":missing_monsters,
            "imagined_monsters": imagined_monsters, "missing_explosions":missing_explosions, "imagined_explosions": imagined_explosions,
            "missing_walls": missing_walls, "imagined_walls":imagined_walls}

def count_events_on_trained_rnn(trained_vae, trained_rnn, initial_latent_vector, actions, num_timesteps = 100):
    assert(len(actions)>=num_timesteps)
    dreamed_latents = []
    dreamed_latent, mixture_weights = trained_rnn.predict_one_step(actions[0], previous_z=initial_latent_vector)
    dreamed_latents.append(dreamed_latent)
    for i in range(num_timesteps-1):
        dreamed_latent, mixture_weights = trained_rnn.predict_one_step(actions[i+1])
        dreamed_latents.append(dreamed_latent)
    predicted_images = trained_vae.decoder.predict(np.array(dreamed_latents))

    return count_events_from_images(predicted_images)

#Measurements indicate that the different mixtures do not indicate different things that could happen (e.g. fireballs appearing,
#monsters disappearing, etc). Rather, they encode different scenarios with different "rules" / laws of physics, such as
#normal scenes (fireballs propagating, etc) and explosion-scenes (where very different rules apply).
def count_explosion_vs_non_explosion_events(trained_vae, trained_rnn, initial_latent_vector, actions, num_timesteps = 100):
    assert(len(actions)>=num_timesteps)
    dreamed_latents = []
    dreamed_latent, mixture_weights = trained_rnn.predict_one_step(actions[0], previous_z=initial_latent_vector)
    dreamed_latents.append(dreamed_latent)
    for i in range(num_timesteps-1):
        dreamed_latent, mixture_weights = trained_rnn.predict_one_step(actions[i+1])
        dreamed_latents.append(dreamed_latent)
    print("Dreamed latents shape is ", np.array(dreamed_latents).shape)
    predicted_images = trained_vae.decoder.predict(np.array(dreamed_latents))

    return count_events_from_images(predicted_images)

def count_appearances_and_disappearances(trained_vae, trained_rnn, initial_latent_vector, actions, num_timesteps = 100):
    assert(len(actions)>=num_timesteps)
    dreamed_latents = []
    dreamed_latent, mixture_weights = trained_rnn.predict_one_step(actions[0], previous_z=initial_latent_vector)
    dreamed_latents.append(dreamed_latent)
    for i in range(num_timesteps-1):
        dreamed_latent, mixture_weights = trained_rnn.predict_one_step(actions[i+1])
        dreamed_latents.append(dreamed_latent)

    predicted_images = trained_vae.decoder.predict(np.array(dreamed_latents))

    return count_appearances_and_disappearances_from_images(predicted_images)

def count_differences_between_reality_and_prediction(trained_vae, trained_rnn, real_latent_sequence, actions):
    #real latent sequences: the N observations. Actions: The N-1 actions BETWEEN those observations.
    assert(len(actions)>= len(real_latent_sequence)-1)
    real_images = trained_vae.decoder.predict(np.array(real_latent_sequence))
    dreamed_latents = []
    for i in range(len(real_latent_sequence)-1):
        dreamed_latent, mixture_weights = trained_rnn.predict_one_step(actions[i], previous_z=real_latent_sequence[i])
        dreamed_latents.append(dreamed_latent)

    dreamed_images = trained_vae.decoder.predict(np.array(dreamed_latents)) #The predictions for the NEXT image after the N-1 first observations.

    #Lining up the predictions with the actual timestep they predict here.
    return count_different_events_in_images(real_images[1:], dreamed_images)
