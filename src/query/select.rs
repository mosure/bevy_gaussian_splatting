use bevy::prelude::*;


#[derive(Component, Debug, Default, Reflect)]
pub struct Select {
    pub indicies: Vec<usize>,
    pub enabled: bool,
}

#[derive(Component, Debug, Default, Reflect)]
pub enum Behavior {
    #[default]
    Hide,
    Show,
}


#[derive(Default)]
pub struct SelectPlugin;

impl Plugin for SelectPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, apply_behaviors);
    }
}


fn apply_behaviors(

) {

}
